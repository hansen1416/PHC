import os
import sys
import re
import unicodedata
from glob import glob
from pathlib import Path
import io
import pickle
import subprocess

sys.path.append(os.getcwd())

import torch
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState


SMPL_MUJOCO_NAMES = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe", "Torso",
    "Spine", "Chest", "Neck", "Head", "L_Thorax",
    "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
]

SMPL_BONE_ORDER_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Torso", "L_Knee", "R_Knee",
    "Spine", "L_Ankle", "R_Ankle", "Chest", "L_Toe", "R_Toe",
    "Neck", "L_Thorax", "R_Thorax", "Head", "L_Shoulder",
    "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
    "L_Hand", "R_Hand",
]

SMPL_2_MUJOCO = [
    SMPL_BONE_ORDER_NAMES.index(q)
    for q in SMPL_MUJOCO_NAMES
    if q in SMPL_BONE_ORDER_NAMES
]

RCLONE_REMOTE_DIR = "gdrive:humos_phc_results"

# cache by (gender, beta_key, device, dtype)
_SKELETON_CACHE = {}


def safe_prefix_filename(text: str, n: int = 24) -> str:
    """
    Take the first n characters of `text`, replace spaces with underscores,
    and sanitize to a filesystem-safe ASCII-ish token.
    """
    if not isinstance(text, str):
        text = str(text)

    s = text[:n]
    s = re.sub(r"\s+", "_", s.strip())
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9._-]", "_", s)
    s = re.sub(r"_+", "_", s).strip("._-")
    return s or "untitled"


def _axis_angle_to_quat_xyzw(aa: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert axis-angle (..., 3) to quaternion (..., 4) in xyzw order.
    Fully torch-based, so it runs on GPU.
    """
    angle = torch.linalg.norm(aa, dim=-1, keepdim=True)
    half = 0.5 * angle

    scale = torch.where(
        angle > eps,
        torch.sin(half) / angle,
        0.5 - (angle * angle) / 48.0,
    )
    xyz = aa * scale
    w = torch.where(
        angle > eps,
        torch.cos(half),
        1.0 - (angle * angle) / 8.0,
    )
    return torch.cat([xyz, w], dim=-1)


def _quat_conjugate_xyzw(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., :3] = -out[..., :3]
    return out


def _quat_mul_xyzw(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """
    Quaternion multiply in xyzw convention.
    """
    x1, y1, z1, w1 = q.unbind(dim=-1)
    x2, y2, z2, w2 = r.unbind(dim=-1)

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return torch.stack([x, y, z, w], dim=-1)


def _move_obj_float_tensors_to_device(obj, device, dtype=None):
    """
    Move only floating-point tensors to GPU.
    Keep integer/index tensors on CPU, because poselib may call .numpy() on them.
    """
    for k, v in obj.__dict__.items():
        if torch.is_tensor(v) and v.is_floating_point():
            if dtype is not None:
                setattr(obj, k, v.to(device=device, dtype=dtype))
            else:
                setattr(obj, k, v.to(device=device))
    return obj


def _get_skeleton_tree(gender: str, beta_key: str, device, dtype=torch.float32):
    key = (gender, beta_key, str(device), str(dtype))
    if key not in _SKELETON_CACHE:
        mjcf_path = os.path.join(
            f"/home/hlz/repos/hhi/ase/data/assets/mjcf/smpl/{gender}_{beta_key}_smpl.xml"
        )
        sk_tree = SkeletonTree.from_mjcf(mjcf_path)
        sk_tree = _move_obj_float_tensors_to_device(sk_tree, device=device, dtype=dtype)
        _SKELETON_CACHE[key] = sk_tree
    return _SKELETON_CACHE[key]


def calc_pose_quat(gender, beta_key, pose_aa, root_trans, device):
    pose_aa = pose_aa.to(device)
    root_trans = root_trans.to(device)

    n = pose_aa.shape[0]

    pose_aa_mj = pose_aa.reshape(n, 24, 3)[:, SMPL_2_MUJOCO]
    pose_quat = _axis_angle_to_quat_xyzw(pose_aa_mj.reshape(-1, 3)).reshape(n, 24, 4)

    skeleton_tree = _get_skeleton_tree(
        gender,
        beta_key,
        device=device,
        dtype=pose_quat.dtype,
    )

    root_trans_offset = root_trans + skeleton_tree.local_translation[0]

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        pose_quat,
        root_trans_offset,
        is_local=True,
    )

    # same upright correction as the old script, but done fully in torch
    upright_q = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=pose_quat.dtype, device=device)
    upright_inv = _quat_conjugate_xyzw(upright_q).view(1, 1, 4)

    pose_quat_global = _quat_mul_xyzw(
        new_sk_state.global_rotation,
        upright_inv,
    )

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        pose_quat_global,
        root_trans_offset,
        is_local=False,
    )

    pose_quat = new_sk_state.local_rotation
    pose_quat_global = new_sk_state.global_rotation

    return root_trans_offset, pose_quat, pose_quat_global

DEFAULT_INPUT_FOLDER = os.path.join("/mnt", "gdrive_humos_output")
DEFAULT_OUTPUT_DIR = os.path.join("/home", "hlz/datasets/humos_results_full")

def data_format_humos2phc(humos_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    humos_result = torch.load(humos_path, map_location=device, weights_only=False)
    motion_id = Path(humos_path).stem

    with torch.inference_mode():
        for gender in ["male", "female"]:
            for beta_key, humos_motion_data in humos_result[gender].items():
                n_frame = humos_motion_data["trans"].shape[0]

                phc_motion = {}

                phc_motion["beta"] = humos_motion_data["betas"][0].to(device)
                phc_motion["trans_orig"] = humos_motion_data["trans"].to(device)

                pose_aa = torch.zeros(
                    n_frame,
                    24,
                    3,
                    device=device,
                    dtype=phc_motion["trans_orig"].dtype,
                )
                pose_aa[:, 0, :] = humos_motion_data["root_orient"].to(device)
                pose_aa[:, 1:, :] = humos_motion_data["pose_body"].to(device)
                phc_motion["pose_aa"] = pose_aa.reshape(n_frame, -1)

                root_trans_offset, pose_quat, pose_quat_global = calc_pose_quat(
                    gender,
                    beta_key,
                    phc_motion["pose_aa"],
                    phc_motion["trans_orig"],
                    device,
                )

                phc_motion["root_trans_offset"] = root_trans_offset
                phc_motion["pose_quat"] = pose_quat
                phc_motion["pose_quat_global"] = pose_quat_global

                # store tensors back on CPU for pickle/rclone output
                for k, v in phc_motion.items():
                    if torch.is_tensor(v):
                        phc_motion[k] = v.detach().to(device="cpu", dtype=torch.float32)

                phc_motion["offset_height"] = humos_motion_data["offset_height"]
                phc_motion["beta_key"] = beta_key
                phc_motion["gender"] = gender
                phc_motion["fps"] = 20

                motion_key = f"{motion_id}_{gender}_{beta_key}"
                output_file = os.path.join(DEFAULT_OUTPUT_DIR, f"{motion_key}.pkl")

                payload = {motion_key: phc_motion}

                if False:
                    save_pkl_local(payload, output_file)

                

def save_pkl_local(obj: object, output_file: str) -> None:
    # os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    from tqdm import tqdm

    pattern = os.path.join(DEFAULT_INPUT_FOLDER, "*.pt")
    files = sorted(glob(pattern, recursive=True))

    files = files[:10000]

    pbar = tqdm(files, desc="t", unit="file")
    for file in pbar:
        pbar.set_postfix_str(os.path.basename(file))
        data_format_humos2phc(file)