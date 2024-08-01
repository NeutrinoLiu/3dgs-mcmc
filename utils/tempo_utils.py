import random
import torch

def rotvec2quat(batched_quat):
    angles = torch.norm(batched_quat, dim=1, keepdim=True)  # Norm of each vector (angle of rotation)
    unit_vectors = torch.nn.functional.normalize(batched_quat)  # Normalized direction vectors

    half_angles = angles / 2
    cos_half_angles = torch.cos(half_angles).squeeze(-1)  # w component
    sin_half_angles = torch.sin(half_angles).squeeze(-1)  # Factor for x, y, z components
    
    quaternions = torch.zeros((batched_quat.shape[0], 4), dtype=batched_quat.dtype, device=batched_quat.device)
    quaternions[:, 0] = cos_half_angles  # w
    quaternions[:, 1:] = unit_vectors * sin_half_angles.unsqueeze(-1)  # x, y, z

    return quaternions

def rotvec2mat(batched_vec):
    '''
    TODO: debug
    '''
    angles = torch.norm(batched_vec, dim=1, keepdim=True)  # Norm of each vector (angle of rotation)
    unit_vectors = torch.nn.functional.normalize(batched_vec)  # Normalized direction vectors
    kx, ky, kz = unit_vectors[:, 0], unit_vectors[:, 1], unit_vectors[:, 2]
    
    K = torch.zeros((batched_vec.shape[0], 3, 3), dtype=batched_vec.dtype, device=batched_vec.device)
    K[:, 0, 1] = -kz
    K[:, 0, 2] = ky
    K[:, 1, 0] = kz
    K[:, 1, 2] = -kx
    K[:, 2, 0] = -ky
    K[:, 2, 1] = kx
    I = torch.eye(3, device=batched_vec.device).unsqueeze(0).repeat(batched_vec.shape[0], 1, 1)  # Identity matrix
    sin_theta = torch.sin(angles).unsqueeze(-1)
    cos_theta = torch.cos(angles).unsqueeze(-1)
    K2 = torch.bmm(K, K)
    
    R = I + sin_theta * K + (1 - cos_theta) * K2
    return R

def quat_mul(batched_q1, batched_q2):
    assert batched_q1.shape == batched_q2.shape, "Both tensors must have the same shape"
    w1, x1, y1, z1 = batched_q1[:, 0], batched_q1[:, 1], batched_q1[:, 2], batched_q1[:, 3]
    w2, x2, y2, z2 = batched_q2[:, 0], batched_q2[:, 1], batched_q2[:, 2], batched_q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((w, x, y, z), dim=1)

def rigid_deform(xyz, rot,
                 rigid_v,
                 rigid_rotvec,
                 rigid_rotcen,
                 time_span,
                 skip=False):
    assert xyz.shape[0] == rigid_v.shape[0] == time_span.shape[0], f"Batch size mismatch: {xyz.shape[0]} != {rigid_v.shape[0]} != {time_span.shape[0]}"
    if skip:
        # avoid empty gradient, you need to use those tensor
        ret_xyz = xyz + rigid_v * 0 + rigid_rotvec * 0 + rigid_rotcen * 0
        return  ret_xyz, rot

    time_span_unsqueeze = time_span.unsqueeze(-1)
    position_shift = rigid_v * time_span_unsqueeze
    rotation_vec = rigid_rotvec * time_span_unsqueeze
    # NOTE: quaternion is in w.xyz format, i.e. scale first
    # check GaussianModel.create_from_pcd()
    rotation_quad = rotvec2quat(rotation_vec)

    # new gaussians position para
    rotation_mat = rotvec2mat(rotation_vec)
    xyz_ret = torch.bmm(rotation_mat,
                        (xyz - rigid_rotcen).unsqueeze(-1)).squeeze(-1)
    xyz_ret += rigid_rotcen + position_shift
    
    # new gaussians rotation para
    # dont have to normalize, it will always get normalized before fed to render
    rot_ret = quat_mul(rotation_quad, rot)

    # print(f"xyz before {xyz[0]}, xyz after {xyz_ret[0]}")
    return xyz_ret, rot_ret

class SliWinManager:
    def __init__(self, win_size, max_frame, max_sample=1):
        self.frame_start = 0
        self.frame_end = win_size
        self.max_frame = max_frame
        self.max_sample = max_sample
        self._sampled_frames = None

    def state_dump(self):
        return {
            "frame_start": self.frame_start,
            "frame_end": self.frame_end,
            "max_frame": self.max_frame,
            "_sampled_frames": self._sampled_frames
        }
    def state_load(self, state_dict):
        self.frame_start = state_dict["frame_start"]
        self.frame_end = state_dict["frame_end"]
        self.max_frame = state_dict["max_frame"]
        self._sampled_frames = state_dict["_sampled_frames"]
    def __str__(self):
        return f"window[{self.frame_start}:{self.frame_end}]"
    def tick(self):
        self.frame_start += 1
        self.frame_end += 1
    def fetch_cams(self, fetcher):
        return fetcher(self.sampled_frames()).copy()
    def sampled_frames(self, resample=True):
        if resample or (self._sampled_frames is None):
            self._sampled_frames = self.all_frames()
            if len(self._sampled_frames) > self.max_sample:
                self._sampled_frames = sorted(random.sample(self._sampled_frames, self.max_sample))
                print(f"Warning: too many frames in window, resample {self.max_sample} from {self}")
                print(f"Sampled frames: {self._sampled_frames}")
        return self._sampled_frames
    def all_frames(self):
        return range(self.frame_start, min(self.frame_end, self.max_frame))

