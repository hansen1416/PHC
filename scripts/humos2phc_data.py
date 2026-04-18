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

import joblib
import torch
from scipy.spatial.transform import Rotation as sRot
from poselib.poselib.skeleton.skeleton3d import SkeletonTree, SkeletonState



SMPL_MUJOCO_NAMES = [
    "Pelvis",
    "L_Hip",
    "L_Knee",
    "L_Ankle",
    "L_Toe",
    "R_Hip",
    "R_Knee",
    "R_Ankle",
    "R_Toe",
    "Torso",
    "Spine",
    "Chest",
    "Neck",
    "Head",
    "L_Thorax",
    "L_Shoulder",
    "L_Elbow",
    "L_Wrist",
    "L_Hand",
    "R_Thorax",
    "R_Shoulder",
    "R_Elbow",
    "R_Wrist",
    "R_Hand",
]


SMPL_BONE_ORDER_NAMES = [
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Torso",
    "L_Knee",
    "R_Knee",
    "Spine",
    "L_Ankle",
    "R_Ankle",
    "Chest",
    "L_Toe",
    "R_Toe",
    "Neck",
    "L_Thorax",
    "R_Thorax",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
]

RCLONE_REMOTE_DIR = "gdrive:humos_phc_results"


def safe_prefix_filename(text: str, n: int = 24) -> str:
    """
    Take the first n characters of `text`, replace spaces with underscores,
    and sanitize to a filesystem-safe ASCII-ish token.
    """
    if not isinstance(text, str):
        text = str(text)

    # take first n chars, replace whitespace runs with single underscore
    s = text[:n]
    s = re.sub(r"\s+", "_", s.strip())

    # normalize to ASCII (drop accents), then keep only safe chars
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9._-]", "_", s)  # replace unsafe chars with _
    s = re.sub(r"_+", "_", s).strip("._-")  # collapse underscores, trim edges

    return s or "untitled"


def calc_pose_quat(gender, beta_key, pose_aa, root_trans, device):

    N = pose_aa.shape[0]

    smpl_2_mujoco = [
        SMPL_BONE_ORDER_NAMES.index(q)
        for q in SMPL_MUJOCO_NAMES
        if q in SMPL_BONE_ORDER_NAMES
    ]
    pose_aa_mj = pose_aa.reshape(N, 24, 3)[:, smpl_2_mujoco]
    pose_quat = sRot.from_rotvec(pose_aa_mj.reshape(-1, 3)).as_quat().reshape(N, 24, 4)

    # print(pose_quat.shape)

    skeleton_tree = SkeletonTree.from_mjcf(
        os.path.join(f"/home/hlz/repos/hhi/ase/data/assets/mjcf/smpl/{gender}_{beta_key}_smpl.xml")
    )

    root_trans_offset = root_trans + skeleton_tree.local_translation[0].to(device)

    new_sk_state = SkeletonState.from_rotation_and_root_translation(
                    skeleton_tree,  # This is the wrong skeleton tree (location wise) here, but it's fine since we only use the parent relationship here. 
                    torch.from_numpy(pose_quat).to(device),
                    root_trans_offset.to(device),
                    is_local=True)
    
    if True:# this upright_start flag
        pose_quat_global = (sRot.from_quat(new_sk_state.global_rotation.reshape(-1, 4).numpy()) * sRot.from_quat([0.5, 0.5, 0.5, 0.5]).inv()).as_quat().reshape(N, -1, 4)  # should fix pose_quat as well here...

        new_sk_state = SkeletonState.from_rotation_and_root_translation(skeleton_tree, torch.from_numpy(pose_quat_global), root_trans_offset, is_local=False)
        pose_quat = new_sk_state.local_rotation.numpy()
    
    pose_quat_global = new_sk_state.global_rotation
    pose_quat = new_sk_state.local_rotation

    # print(pose_quat_global.shape, pose_quat.shape)

    return root_trans_offset, pose_quat, pose_quat_global



def data_format_humos2phc(humos_path):

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cpu"

    humos_result = torch.load(humos_path, map_location=device)

    motion_id = Path(humos_path).stem

    for gender in ["male", "female"]:

        for beta_key, humos_motion_data in humos_result[gender].items():

            n_frame = humos_motion_data["trans"].shape[0]

            phc_motion = {}

            phc_motion['beta'] = humos_motion_data["betas"][0]

            # [n, 3]
            phc_motion["trans_orig"] = humos_motion_data["trans"]
            # [n, 24, 3]
            phc_motion["pose_aa"] = torch.zeros(n_frame, 24, 3)

            phc_motion["pose_aa"][:, 0, :] = humos_motion_data["root_orient"]
            phc_motion["pose_aa"][:, 1:, :] = humos_motion_data["pose_body"]

            phc_motion["pose_aa"] = phc_motion["pose_aa"].reshape(n_frame, -1)

            root_trans_offset, pose_quat, pose_quat_global = calc_pose_quat(gender, beta_key, phc_motion["pose_aa"], phc_motion["trans_orig"], device)

            phc_motion['root_trans_offset'] = root_trans_offset
            phc_motion["pose_quat"] = pose_quat
            phc_motion["pose_quat_global"] = pose_quat_global

            for k, v in phc_motion.items():
                phc_motion[k] = v.to(torch.float32)
            
            phc_motion["offset_height"] = humos_motion_data["offset_height"]
            phc_motion['beta_key'] = beta_key
            phc_motion["gender"] = gender
            phc_motion['fps'] = 20

            motion_key = f"{motion_id}_{gender}_{beta_key}"
            # file_path = os.path.join(output_dir, f"{motion_key}.pkl")

            # print(f"dumping {file_path}")

            # joblib.dump(
            #     {f"{motion_key}": phc_motion},
            #     file_path
            # )

            remote_file = f"{RCLONE_REMOTE_DIR}/{motion_key}.pkl"

            upload_pkl_with_rclone({f"{motion_key}": phc_motion}, remote_file)


def upload_pkl_with_rclone(obj, remote_file):
    buf = io.BytesIO()
    pickle.dump(obj, buf, protocol=pickle.HIGHEST_PROTOCOL)
    data = buf.getvalue()

    subprocess.run(
        ["rclone", "rcat", "--retries", "3", remote_file],
        input=data,
        check=True,
    )

if __name__ == "__main__":

    from tqdm import tqdm

    folder = os.path.join(
        os.path.expanduser("~"),
        "repos/humos/output",
    )

    pattern = os.path.join(folder, "**", f"*.pt")
    files = glob(pattern, recursive=True)
    files = sorted(files)

    pbar = tqdm(files, desc="t", unit="file")
    for file in pbar:
        pbar.set_postfix_str(os.path.basename(file))
        data_format_humos2phc(file)
