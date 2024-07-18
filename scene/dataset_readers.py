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
import sys
import random
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    extra_para: dict = None
    frame: int = -1

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

class DynamicSceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cam_at: list
    test_cam_at: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8, init_type="sfm", num_pts=100000):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if init_type == "sfm":
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
    elif init_type == "random":
        ply_path = os.path.join(path, "random.ply")
        print(f"Generating random point cloud ({num_pts})...")
        
        xyz = np.random.random((num_pts, 3)) * nerf_normalization["radius"]* 3*2 -(nerf_normalization["radius"]*3)
        
        num_pts = xyz.shape[0]
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        print("Please specify a correct init_type: random or sfm")
        exit(0)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

# ----------------------- google immersive scene loader ---------------------- #
def readGoogleCameras(cams: dict, images_folder):
    cam_infos = []
    for cam_name, paras in cams.items():
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}".format(cam_name))
        sys.stdout.flush()

        extr = paras['extrinsic']
        intr = paras['intrinsic']
        height = intr['height']
        width = intr['width']

        uid = int(cam_name.split('_')[-1].split('Cam')[-1].split('.')[0])

        R = np.array(extr['SO3']).T
        T = np.array(extr['T'])

        focal_length_x = intr['matrix'][0][0]
        focal_length_y = intr['matrix'][1][1]
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
        cx, cy = intr['matrix'][0][-1], intr['matrix'][1][-1] 
        extra = {
            "cx": cx,
            "cy": cy,
            "focal_x": focal_length_x,
            "focal_y": focal_length_y
        }
        # extra=None

        image_name = f"{cam_name}"
        image_path = os.path.join(images_folder, image_name)
        image = Image.open(image_path)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              extra_para=extra)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readGoogleImmersiveInfo(path, images, eval, llffhold=8, init_type="random", num_pts=100000):
    
    cameras_file = os.path.join(path, "cam.json")
    with open(cameras_file, "r") as f:
        cams_para = json.load(f)

    reading_dir = "undistorted"

    cam_infos_unsorted = readGoogleCameras(cams=cams_para, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    # --------------------------- random only ply init --------------------------- #
    ply_path = os.path.join(path, "random.ply")
    print(f"Generating random point cloud ({num_pts})...")
    
    xyz = np.random.random((num_pts, 3)) * nerf_normalization["radius"]* 3*2 -(nerf_normalization["radius"]*3)
    
    num_pts = xyz.shape[0]
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    storePly(ply_path, xyz, SH2RGB(shs) * 255)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

# ----------------------------- SwinGS dataloader ---------------------------- #
def readFixedCams(cams: dict):
    '''
    dont read img at this time
    '''
    cam_infos = []
    for cam_name, paras in cams.items():
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}".format(cam_name))
        sys.stdout.flush()

        extr = paras['extrinsic']
        intr = paras['intrinsic']
        height = intr['height']
        width = intr['width']

        uid = int(cam_name.split('_')[-1].split('Cam')[-1].split('.')[0])

        R = np.array(extr['SO3']).T
        T = np.array(extr['T'])

        focal_length_x = intr['matrix'][0][0]
        focal_length_y = intr['matrix'][1][1]
        FovY = focal2fov(focal_length_y, height)
        FovX = focal2fov(focal_length_x, width)
        cx, cy = intr['matrix'][0][-1], intr['matrix'][1][-1] 
        extra = {
            "cx": cx,
            "cy": cy,
            "focal_x": focal_length_x,
            "focal_y": focal_length_y
        }
        # extra=None

        image_name = f"{cam_name}"
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=None,
                              image_path=None, image_name=image_name, width=width, height=height,
                              extra_para=extra)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readDynamicSceneInfo(path, images, eval, llffhold=8, init_type="random", num_pts=100000,
                         max_frame=10, tempo_shuffle=False):
    '''
    a SwinGS dataset expect to follow the following dir structure:

        - dataset_name
            - images_per_frame
                - 0
                - 1
                ...
            - cam.json

    each subdir in images_per_frame should includes all images shooted by 
    every camera mentioned in cam.json
    '''
    cameras_file = os.path.join(path, "cam.json")
    with open(cameras_file, "r") as f:
        cams_para = json.load(f)

    reading_dir = "images_per_frame"
    for t in range(max_frame):
        max_frame_dir = os.path.join(path, reading_dir, str(t))
        assert os.path.exists(max_frame_dir), f"missing frame dir: {max_frame_dir}"

    fixed_cam_infos_unsorted = readFixedCams(cams=cams_para)
    print([c.image_name for c in fixed_cam_infos_unsorted])
    fixed_cam_infos = sorted(fixed_cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_test_split = fixed_cam_infos.copy()
    train_cam_info_at = []
    test_cam_info_at = []

    def update_img_path(c, t, load=False):
        uid = f"{t}.{c.uid}"
        image_name = os.path.join(str(t), c.image_name)
        image_path = os.path.join(path, reading_dir, image_name)
        image = Image.open(image_path) if load else None

        return CameraInfo(  uid=uid, 
                            R=c.R, T=c.T, FovY=c.FovY, FovX=c.FovX, 
                            image=image, image_path=image_path, image_name=image_name,
                            width=c.width, height=c.height, extra_para=c.extra_para,
                            frame = t)
    
    if eval:
        if tempo_shuffle: random.seed(42)
        for t in range(max_frame):
            trains_at_t = []
            tests_at_t = []
            if tempo_shuffle: random.shuffle(train_test_split)
            for idx, c in enumerate(train_test_split):
                image_at_t = update_img_path(c, t)
                if idx % llffhold != 0:
                    trains_at_t.append(image_at_t)
                else:
                    tests_at_t.append(image_at_t)
            train_cam_info_at.append(trains_at_t)
            test_cam_info_at.append(tests_at_t)
    else:
        for t in range(max_frame):
            train_cam_info_at.append([update_img_path(c, t) for c in train_test_split])
            test_cam_info_at.append([])
    
    # use the first frame camera to init random points
    nerf_normalization = getNerfppNorm(train_cam_info_at[0]) 

    # --------------------------- random only ply init --------------------------- #
    ply_path = os.path.join(path, "random.ply")
    print(f"Generating random point cloud ({num_pts})...")
    
    xyz = np.random.random((num_pts, 3)) * nerf_normalization["radius"]* 3*2 -(nerf_normalization["radius"]*3)
    
    num_pts = xyz.shape[0]
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    storePly(ply_path, xyz, SH2RGB(shs) * 255)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = DynamicSceneInfo(point_cloud=pcd,
                           train_cam_at = train_cam_info_at,
                           test_cam_at = test_cam_info_at,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Google" : readGoogleImmersiveInfo,
    "SwinGS": readDynamicSceneInfo
}

