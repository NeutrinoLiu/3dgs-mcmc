#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from PIL import Image
from utils.general_utils import PILtoTorch
from torch import nn
import os
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixShift

class Camera(nn.Module):
    def load(self):
        pass
    def unload(self):
        pass
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                 extra_para=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        if extra_para is not None:
            # print(extra_para)
            self.projection_matrix = getProjectionMatrixShift(znear=self.znear, zfar=self.zfar, focal_x=extra_para["focal_x"], focal_y=extra_para["focal_y"], 
                                                          cx=extra_para["cx"], cy=extra_para["cy"], width=self.image_width, height=self.image_height,
                                                          fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        else:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

class LazyCamera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda",
                  # new member fields
                 frame=0,
                 extra_para=None,
                 resolution_scale=1.0,
                 args_resolution=-1,
                 image_path=None
                 ):
        super(LazyCamera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        assert os.path.exists(image_path), f"missing image {image_path}"
        self.extra_para = extra_para
        self.image_path = image_path
        self.resolution_scale = resolution_scale
        self.args_resolution = args_resolution

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")


        self.frame = frame

        self.original_image = None
        self.image_width = None
        self.image_height = None

        self.world_view_transform = None
        self.projection_matrix = None
        self.full_proj_transform = None
        self.camera_center = None
    
    def load(self): 
        '''
        this is a copy of camera_utils/loadCam() for lazy loading
        '''
        # ------------------------------- loadCam part ------------------------------- #
        if self.original_image is not None:
            print(f"duplicate loading cam {self.image_name}")
            return 
        image = Image.open(self.image_path)
        resolution_scale = self.resolution_scale
        args_resolution = self.args_resolution # args.resolution

        orig_w, orig_h = image.size

        if args_resolution in [1, 2, 4, 8]:
            resolution = round(orig_w/(resolution_scale * args_resolution)), round(orig_h/(resolution_scale * args_resolution))
        else:  # should be a type that converts to float
            if args_resolution == -1:
                if orig_w > 1600:
                    global WARNED
                    if not WARNED:
                        print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                            "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                        WARNED = True
                    global_down = orig_w / 1600
                else:
                    global_down = 1
            else:
                global_down = orig_w / args_resolution

            scale = float(global_down) * float(resolution_scale)
            resolution = (int(orig_w / scale), int(orig_h / scale))

        resized_image_rgb = PILtoTorch(image, resolution)

        post_resized_image = resized_image_rgb[:3, ...]
        gt_alpha_mask = None
        
        if resized_image_rgb.shape[1] == 4:
            gt_alpha_mask = resized_image_rgb[3:4, ...]

        # ----------------------------- camera init part ----------------------------- #
        self.original_image = post_resized_image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, self.trans, scale)).transpose(0, 1).cuda()
        if self.extra_para is not None:
            self.projection_matrix = getProjectionMatrixShift(znear=self.znear, zfar=self.zfar, focal_x=self.extra_para["focal_x"], focal_y=self.extra_para["focal_y"], 
                                                          cx=self.extra_para["cx"], cy=self.extra_para["cy"], width=self.image_width, height=self.image_height,
                                                          fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        else:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        # dont need this image in memory
        image.close()

    def unload(self):
        del self.original_image
        del self.image_width
        del self.image_height
        del self.world_view_transform
        del self.projection_matrix
        del self.full_proj_transform
        del self.camera_center

        self.original_image = None              # on cuda
        self.image_width = None
        self.image_height = None
        self.world_view_transform = None        # on cuda
        self.projection_matrix = None           # on cuda
        self.full_proj_transform = None         # on cuda
        self.camera_center = None               # on cuda