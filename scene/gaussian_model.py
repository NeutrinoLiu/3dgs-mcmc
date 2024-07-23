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


# ------- a modified 3dgs supporting deformation within a short window ------- #
# ----------------------------- bliu277@wisc.edu ----------------------------- #

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.reloc_utils import compute_relocation_cuda
from utils.tempo_utils import deform

def indices_of(tensor):
    '''
    return the index of non-zero elements in the tensor
    '''
    return torch.nonzero(tensor.squeeze(-1) , as_tuple=True)[0]

class GaussianModel:

    def setup_functions(self):
        '''
        activations to ensure data range
        '''
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree : int, max_lifespan : int):
        '''
        attributes
        '''
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.max_lifespan = max_lifespan

        self.setup_functions()

        # immatured gaussians, gradient is needed here
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)

        self._frame_birth = torch.empty(0)
        self._frame_start = torch.empty(0)
        self._frame_end = torch.empty(0)

        '''
        spatial model
        currently we use a native circular motion model,
        maybe we could use quarterion for temporal rotation in the future
        '''
        # self._drift_v = torch.empty(0) # float(3) linear volocity.
        # self._drift_w = torch.empty(0) # float(1) angular volocity
        # self._drift_o = torch.empty(0) # float(3) center of circular motion
        # self._drift_n = torch.empty(0) # float(3) norm of circular motion plane


        '''
        matured gaussians, no gradient is needed here
        '''
        self._matured_xyz = torch.empty(0).cuda()
        self._matured_features_dc = torch.empty(0).cuda()
        self._matured_features_rest = torch.empty(0).cuda()
        self._matured_scaling = torch.empty(0).cuda()
        self._matured_rotation = torch.empty(0).cuda()
        self._matured_opacity = torch.empty(0).cuda()

        self._matured_frame_birth = torch.empty(0).cuda()
        self._matured_frame_start = torch.empty(0).cuda()
        self._matured_frame_end = torch.empty(0).cuda()

    def capture(self):
        '''
        TODO
        training context snapthot
        dump
        '''
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        '''
        TODO
        training context snapshot
        load
        '''
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        raise NotImplementedError
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        raise NotImplementedError
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        raise NotImplementedError
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        raise NotImplementedError
        return self.opacity_activation(self._opacity)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        '''
        init from colmap SfM key points
        '''
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2)*0.1)[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.5 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # init from colmap SfM key points, use full lifespan
        self._frame_birth = torch.zeros((self.get_xyz.shape[0]), device="cuda", dtype=torch.int32)
        self._frame_start = torch.zeros((self.get_xyz.shape[0]), device="cuda", dtype=torch.int32)
        self._frame_end   = torch.full((self.get_xyz.shape[0],), self.max_lifespan, device="cuda", dtype=torch.int32)

    def training_setup(self, training_args):
        '''
        learning rate & optimizor setup
        '''
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        '''
        TODO
        '''
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        '''
        TODO
        gaussians point cloud only
        dump
        '''
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        '''
        TODO
        gaussians point cloud only
        load
        '''
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    # ^----------------------------- status related ------------------------------ #

    # -------------------------- sliding window related -------------------------- #
    def decay_genesis(self):
        # divide all gs into N groups, randomly
        # each group reduce the frame_end by i, where i in [0, N-1]
        num_of_groups = self.max_lifespan
        assert self._frame_end.shape[0] % num_of_groups == 0, \
            "The number of gaussians should be divisible by number of grouping (i.e. max_lifespan)"
        assert self._frame_end.shape[0] == self._opacity.shape[0], \
            "The number of gaussians should be the same in opacity and frame_end"
        per_group_size = self._frame_end.shape[0] // num_of_groups

        # high opacity genessis gaussian deserves longer lifespan
        _, indices = torch.sort(self._opacity.squeeze(-1), descending=True)
        # indices = torch.randperm(self._xyz.shape[0], device="cuda")
        for i in range(num_of_groups):
            start_idx = i * per_group_size
            group_indices = indices[start_idx : start_idx + per_group_size]
            self._frame_end[group_indices] -= i

    def _mature(self, mature_idx):
        '''
        move gaussians from immatured to matured
        '''
        len_before = self._matured_xyz.shape[0]

        self._matured_xyz = torch.cat((self._matured_xyz, self._xyz[mature_idx]), dim=0).detach()
        self._matured_features_dc = torch.cat((self._matured_features_dc, self._features_dc[mature_idx]), dim=0).detach()
        self._matured_features_rest = torch.cat((self._matured_features_rest, self._features_rest[mature_idx]), dim=0).detach()
        self._matured_scaling = torch.cat((self._matured_scaling, self._scaling[mature_idx]), dim=0).detach()
        self._matured_rotation = torch.cat((self._matured_rotation, self._rotation[mature_idx]), dim=0).detach()
        self._matured_opacity = torch.cat((self._matured_opacity, self._opacity[mature_idx]), dim=0).detach()

        self._matured_frame_birth = torch.cat((self._matured_frame_birth, self._frame_birth[mature_idx]), dim=0).detach()
        self._matured_frame_start = torch.cat((self._matured_frame_start, self._frame_start[mature_idx]), dim=0).detach()
        self._matured_frame_end = torch.cat((self._matured_frame_end, self._frame_end[mature_idx]), dim=0).detach()

        len_after = self._matured_xyz.shape[0]
        print("Matured {} gaussians, total {} now".format(len_after - len_before, len_after))

    def _rollover(self, mature_idx, elapse):

        # we need a dynamic model to update the motion related paras
        # self._xyz = self._xyz
        # self._features_dc = self._features_dc
        # self._features_rest = self._features_rest
        # self._scaling = self._scaling
        # self._rotation = self._rotation
        # self._opacity = self._opacity

        self._frame_birth[mature_idx] = self._frame_end[mature_idx]
        self._frame_start[mature_idx] = self._frame_birth[mature_idx]
        self._frame_end[mature_idx] += elapse

    def evolve(self, swin_mgr):
        # find out those immatured gaussians who canot fill up the whole window
        # - reproduce them
        # - mature them
        # NOTE: we have a constraint that MAX_LIFESPAN == SWIN_SIZE
        #       so we can safely assume that gassians need to be matured
        #       are those who can not fill up the whole window
        
        # strictly smaller than sliding window
        mature_idx = indices_of(self._frame_end < swin_mgr.frame_end)

        # the acutual practice could be such that
        #  we snapshot the matured gaussians
        #  and then expand their lifespan 
        #           update their motion related paras
        #  so that they are the next generation now

        self._mature(mature_idx)
        self._rollover(mature_idx, self.max_lifespan)

    def get_immature_para(self, para=["xyz", "feature", "opacity", "scaling", "rotation",
                                      "start_frame", "end_frame", "birth_frame"]):
        '''
        return a dict of gs paras for all immuture gaussians
        '''
        ret = {}
        for p in set(para):
            if p == "xyz":
                ret[p] = self._xyz
            elif p == "feature":
                ret[p] = torch.cat((self._features_dc, self._features_rest), dim=1)
            elif p == "opacity":
                ret[p] = self.opacity_activation(self._opacity)
            elif p == "scaling":
                ret[p] = self.scaling_activation(self._scaling)
            elif p == "rotation":
                ret[p] = self.rotation_activation(self._rotation)
            elif p == "start_frame":
                ret[p] = self._frame_start
            elif p == "end_frame":
                ret[p] = self._frame_end
            elif p == "birth_frame":
                ret[p] = self._frame_birth
            else:
                assert False, "Unknown parameter {}".format(p)
        return ret

    def get_basic_para_at(self, frame, para=["xyz", "feature", "opacity", "scaling", "rotation"]):
        '''
        return a dict of gs paras given a frame
        '''
        ret = {}
        immature_frame_idx = indices_of((self._frame_start <= frame) & (self._frame_end > frame))
        matured_frame_idx = indices_of((self._matured_frame_start <= frame) & (self._matured_frame_end > frame))
        for p in set(para):
            if p == "xyz":
                # TODO: deform the xyz
                ret[p] = torch.cat((self._xyz[immature_frame_idx], self._matured_xyz[matured_frame_idx]), dim=0)
            elif p == "feature":
                im_feature = torch.cat((self._features_dc[immature_frame_idx],
                                       self._features_rest[immature_frame_idx]), dim=1)
                ma_feature = torch.cat((self._matured_features_dc[matured_frame_idx],
                                       self._matured_features_rest[matured_frame_idx]), dim=1)
                ret[p] = torch.cat((im_feature, ma_feature), dim=0)
            elif p == "opacity":
                ret[p] = self.opacity_activation(torch.cat((self._opacity[immature_frame_idx],
                                                            self._matured_opacity[matured_frame_idx]), dim=0))
            elif p == "scaling":
                ret[p] = self.scaling_activation(torch.cat((self._scaling[immature_frame_idx],
                                                            self._matured_scaling[matured_frame_idx]), dim=0))
            elif p == "rotation":
                # TODO: deform the rotation
                ret[p] = self.rotation_activation(torch.cat((self._rotation[immature_frame_idx],
                                                             self._matured_rotation[matured_frame_idx]), dim=0))
            else:
                assert False, "Unknown parameter {}".format(p)
        return ret

    # v---------------------------- optimize related ----------------------------- #


    def cat_tensors_to_optimizer(self, tensors_dict):
        '''
        pretty similar to what replace_tensors_to_optimizer dose,
        but it only extend para list and set NEW guassians' gradiant momentum to zero
        '''
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, reset_params=True):
        '''
        para linked to optimizer get longer
        update the reference in guassianModel object as well
        '''
        
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        if reset_params:
            self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
            self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def replace_tensors_to_optimizer(self, inds=None):
        '''
        basically do the same thing as what cat_tensors_to_optimizer() does,
        but it further set the ORIGINAL copied guassians' gradiant momentum to zero
        '''
        tensors_dict = {"xyz": self._xyz,
            "f_dc": self._features_dc,
            "f_rest": self._features_rest,
            "opacity": self._opacity,
            "scaling" : self._scaling,
            "rotation" : self._rotation}

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            
            if inds is not None:
                stored_state["exp_avg"][inds] = 0
                stored_state["exp_avg_sq"][inds] = 0
            else:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

            del self.optimizer.state[group['params'][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            self.optimizer.state[group['params'][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"] 

        return optimizable_tensors

    def _update_params(self, idxs, ratio):
        '''
        break those chosen gaussians into multiple weaker gaussians
        return a list of REFERENCEs to new values of new gaussians
        '''
        para = self.get_immature_para(para=["opacity", "scaling"])
        new_opacity, new_scaling = compute_relocation_cuda(
            opacity_old=para['opacity'][idxs, 0],
            scale_old=para['scaling'][idxs],
            N=ratio[idxs, 0] + 1
        )
        new_opacity = torch.clamp(new_opacity.unsqueeze(-1), max=1.0 - torch.finfo(torch.float32).eps, min=0.005)
        new_opacity = self.inverse_opacity_activation(new_opacity)
        new_scaling = self.scaling_inverse_activation(new_scaling.reshape(-1, 3))

        return self._xyz[idxs], self._features_dc[idxs], self._features_rest[idxs], new_opacity, new_scaling, self._rotation[idxs]


    def _sample_alives(self, probs, num, alive_indices=None):
        '''
        find the candidate where new/dead gaussians will warp to
        return either a list of index, or a ctr dict
        '''
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs).unsqueeze(-1)
        return sampled_idxs, ratio

    def relocate_gs(self, dead_mask=None):
        '''
        move dead gaussian to those alives
        '''
        if dead_mask.sum() == 0:
            return

        alive_mask = ~dead_mask 
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        if alive_indices.shape[0] <= 0:
            return

        # sample from alive ones based on opacity
        probs = (self.get_opacity[alive_indices, 0]) 
        reinit_idx, ratio = self._sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0])

        (
            self._xyz[dead_indices], 
            self._features_dc[dead_indices],
            self._features_rest[dead_indices],
            self._opacity[dead_indices],
            self._scaling[dead_indices],
            self._rotation[dead_indices] 
        ) = self._update_params(reinit_idx, ratio=ratio)
        
        self._opacity[reinit_idx] = self._opacity[dead_indices]
        self._scaling[reinit_idx] = self._scaling[dead_indices]

        self.replace_tensors_to_optimizer(inds=reinit_idx) 

    def add_new_gs(self, cap_max):
        '''
        increase the number of gaussian by 5% each step until cap_max
        '''
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)

        if num_gs <= 0:
            return 0

        probs = self.get_opacity.squeeze(-1) 

        # idx (index of templates): a list indicates index of those to be copied, 
        # ratio (ctr of templates's showup): a counter indicates the number of copy for those who is at the same index
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)

        (
            new_xyz, 
            new_features_dc,
            new_features_rest,
            new_opacity,
            new_scaling,
            new_rotation 
        ) = self._update_params(add_idx, ratio=ratio)

        # original high opacity gaussian get weakened first
        self._opacity[add_idx] = new_opacity
        self._scaling[add_idx] = new_scaling

        # then add those new gaussians to the end of para list
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, reset_params=False)
        self.replace_tensors_to_optimizer(inds=add_idx)

        return num_gs

    def relocate_gs_immuture(self, dead_mask, alive_mask):
        '''
        move dead gaussian to those alives
        '''
        if dead_mask.sum() == 0 or alive_mask.sum() == 0:
            return

        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        # sample from alive ones based on opacity
        op = self.get_immature_para(para=["opacity"])["opacity"]
        probs = (op[alive_indices, 0]) 
        reinit_idx, ratio = self._sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0])

        (
            self._xyz[dead_indices], 
            self._features_dc[dead_indices],
            self._features_rest[dead_indices],
            self._opacity[dead_indices],
            self._scaling[dead_indices],
            self._rotation[dead_indices] 
        ) = self._update_params(reinit_idx, ratio=ratio)
        
        self._opacity[reinit_idx] = self._opacity[dead_indices]
        self._scaling[reinit_idx] = self._scaling[dead_indices]

        self.replace_tensors_to_optimizer(inds=reinit_idx)

        viable = (self._frame_birth[dead_indices] <= self._frame_birth[reinit_idx]).unsqueeze(-1)
        assert torch.all(viable), "The gaussians to be relocated should born earlier"
        self._frame_start[dead_indices] = self._frame_start[reinit_idx]