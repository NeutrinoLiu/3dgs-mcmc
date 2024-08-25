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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import SwinGaussianModel
from scene.gaussian_model_static import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, cameraList_from_camInfos_lazy

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "cam.json")) and \
            os.path.exists(os.path.join(args.source_path, "undistorted")):
            print("Found cam.json file, assuming Google Immersive data set!")
            scene_info = sceneLoadTypeCallbacks["Google"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, init_type=args.init_type)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file, indent=4)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

class DynamicScene:

    gaussians : SwinGaussianModel
    # modify these values to change the maximum number of frames in memory
    # for a 4090 with 24GB of memory, 10 frames is a good value
    MAX_FRAME_IN_MEMORY = 10
    MAX_TEST_FRAME_IN_MEMORY = 40

    def __init__(self, args : ModelParams, gaussians : SwinGaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        self.activated_train_frame_scale = set()
        self.activated_test_frame_scale = set()

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cam_at = []
        self.test_cam_at = []

        # ----------------------- load cameras into scene_info ----------------------- #
        if os.path.exists(os.path.join(args.source_path, "cam.json")) and \
            os.path.exists(os.path.join(args.source_path, "images_per_frame")):
            print("Found cam.json file, assuming SwinGS data set!")
            scene_info = sceneLoadTypeCallbacks["SwinGS"](args.source_path, args.images, args.eval, 
                                                            max_frame=args.max_frame,
                                                            num_pts=args.init_pts,)
        else:
            assert False, "Could not recognize scene type!"

        # ------------------------------- dump all cams ------------------------------ #
        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cam_at:
                for t in range(len(scene_info.test_cam_at)):
                    camlist.extend(scene_info.test_cam_at[t])
            if scene_info.train_cam_at:
                for t in range(len(scene_info.train_cam_at)):
                    camlist.extend(scene_info.train_cam_at[t])
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file, indent=4)

        assert len(scene_info.test_cam_at) == len(scene_info.train_cam_at), "time length of test cam is different from train"
        self.max_frame = len(scene_info.test_cam_at)
        
        if shuffle:
            for t in range(self.max_frame):
                random.shuffle(scene_info.train_cam_at[t])  # Multi-res consistent random shuffling
                random.shuffle(scene_info.test_cam_at[t])  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        # --------------- load cams in scene_info to actual cam objects -------------- #
        print(f"PRE-loading Cameras for {self.max_frame} frames, with dataset scales {resolution_scales}...")
        for t in range(self.max_frame):
            for resolution_scale in resolution_scales:
                train_cams = {}
                # print(f"PRE-loading Train Cameras @ frame {t}")
                train_cams[resolution_scale] = cameraList_from_camInfos_lazy(scene_info.train_cam_at[t], resolution_scale, args)
                test_cams = {}
                # print(f"PRE-loading Test Cameras  @ frame {t}")
                test_cams[resolution_scale] = cameraList_from_camInfos_lazy(scene_info.test_cam_at[t], resolution_scale, args)
            self.train_cam_at.append(train_cams)
            self.test_cam_at.append(test_cams)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
        # -------------------- init point cloud using init points -------------------- #
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    # single cam fetch
    def getTrainCamerasAt(self, t, scale=1.0):
        return self.batchGetTrainCam([t], scale)[t]
    def getTestCamerasAt(self, t, scale=1.0):
        return self.batchGetTestCam([t], scale)[t]

    def batchGetTrainCam(self, t_list, scale=1.0):
        if len(t_list) > self.MAX_FRAME_IN_MEMORY:
            assert False, f"too many frames to activate, {len(t_list)} > {self.MAX_FRAME_IN_MEMORY}"
        # deactive all frames that are not in t_list
        ts_list = set([(t, scale) for t in t_list])
        cur_list = set(self.activated_train_frame_scale)
        deactive = cur_list - ts_list
        for t, s in deactive:
            self._deactivate(t, s, test=False)
        ret = []
        for t, s in ts_list:
            ret += self._activate(t, s, test=False)
        return ret
    def batchGetTestCam(self, t_list, scale=1.0):
        if len(t_list) > self.MAX_TEST_FRAME_IN_MEMORY:
            assert False, f"too many frames to activate, {len(t_list)} > {self.MAX_TEST_FRAME_IN_MEMORY}"
        # deactive all frames that are not in t_list
        ts_list = set([(t, scale) for t in t_list])
        cur_list = set(self.activated_test_frame_scale)
        deactive = cur_list - ts_list
        for t, s in deactive:
            self._deactivate(t, s, test=True)
        ret = []
        for t, s in ts_list:
            ret += self._activate(t, s, test=True)
        return ret

    # unload all frames
    def unloadAllFrames(self):
        for fands in self.activated_train_frame_scale.copy():
            frame, scale = fands
            self._deactivate(frame, scale, test=False)
        for fands in self.activated_test_frame_scale.copy():
            frame, scale = fands
            self._deactivate(frame, scale, test=True)

    def _activate(self, t, scale, test):
        if test:
            cam_register = self.activated_test_frame_scale
            cam_list = self.test_cam_at
            tag = "Test"
        else:
            cam_register = self.activated_train_frame_scale
            cam_list = self.train_cam_at
            tag = "Train"
        assert t >=0 and t < len(cam_list), f"invalid time frame {t}"

        hash = (t, scale)
        if hash in cam_register:
            return cam_list[t][scale]
        
        for c in cam_list[t][scale]:
            c.load()
        print(f" + Activate {tag} cameras @ frame {t}, scale {scale}")
        cam_register.add(hash)

        return cam_list[t][scale]

    def _deactivate(self, t, scale, test):
        if test:
            cam_register = self.activated_test_frame_scale
            cam_list = self.test_cam_at
            tag = "Test"
        else:
            cam_register = self.activated_train_frame_scale
            cam_list = self.train_cam_at
            tag = "Train"
        assert t >=0 and t < len(cam_list), f"invalid time frame {t}"

        hash = (t, scale)
        if hash not in cam_register:
            print(f" - Deactivating an inactive {tag} cameras @ frame {t}, scale {scale}")
            return
    
        for c in cam_list[t][scale]:
            c.unload()
        cam_register.remove(hash)
        print(f" - Deactivate {tag} cameras @ frame {t}, scale {scale}")

