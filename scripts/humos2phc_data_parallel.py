import argparse
import io
import os
import pickle
import re
import subprocess
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.append(os.getcwd())

import torch
from poselib.poselib.skeleton.skeleton3d import SkeletonState, SkeletonTree

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

DEFAULT_REMOTE_DIR = "gdrive:humos_phc_results"
DEFAULT_INPUT_FOLDER = os.path.join(os.path.expanduser("~"), "repos/humos/output")
DEFAULT_ASSET_ROOT = os.path.join(
    os.path.expanduser("~"), "repos/hhi/ase/data/assets"
)

# Per-process cache.
_SKELETON_TREE_CACHE: Dict[Tuple[str, str, str], SkeletonTree] = {}
_SMPL_TO_MUJOCO = [
    SMPL_BONE_ORDER_NAMES.index(q)
    for q in SMPL_MUJOCO_NAMES
    if q in SMPL_BONE_ORDER_NAMES
]
_UPRIGHT_QUAT_INV = torch.tensor([-0.5, -0.5, -0.5, 0.5], dtype=torch.float32)


def safe_prefix_filename(text: str, n: int = 24) -> str:
    if not isinstance(text, str):
        text = str(text)
    s = text[:n]
    s = re.sub(r"\s+", "_", s.strip())
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9._-]", "_", s)
    s = re.sub(r"_+", "_", s).strip("._-")
    return s or "untitled"


def get_skeleton_tree(gender: str, beta_key: str, asset_root: str) -> SkeletonTree:
    key = (gender, beta_key, asset_root)
    if key not in _SKELETON_TREE_CACHE:
        xml_path = os.path.join(
            asset_root, "mjcf", "smpl", f"{gender}_{beta_key}_smpl.xml"
        )
        _SKELETON_TREE_CACHE[key] = SkeletonTree.from_mjcf(xml_path)
    return _SKELETON_TREE_CACHE[key]


def axis_angle_to_quat(rotvec: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle vectors to quaternions in xyzw order."""
    # rotvec: (..., 3)
    angles = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    half_angles = 0.5 * angles

    # Stable sin(x) / x around zero.
    small = angles < 1e-8
    scale = torch.empty_like(angles)
    scale[~small] = torch.sin(half_angles[~small]) / angles[~small]
    # Taylor: sin(theta/2)/theta ~= 1/2 - theta^2/48
    scale[small] = 0.5 - (angles[small] * angles[small]) / 48.0

    xyz = rotvec * scale
    w = torch.cos(half_angles)
    quat = torch.cat([xyz, w], dim=-1)
    return quat


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    xyz = -q[..., :3]
    w = q[..., 3:4]
    return torch.cat([xyz, w], dim=-1)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion multiply in xyzw order."""
    x1, y1, z1, w1 = q1.unbind(dim=-1)
    x2, y2, z2, w2 = q2.unbind(dim=-1)

    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return torch.stack([x, y, z, w], dim=-1)


def calc_pose_quat(
    gender: str,
    beta_key: str,
    pose_aa: torch.Tensor,
    root_trans: torch.Tensor,
    device: str,
    asset_root: str,
):
    n_frames = pose_aa.shape[0]
    pose_aa_mj = pose_aa.reshape(n_frames, 24, 3)[:, _SMPL_TO_MUJOCO]
    pose_quat = axis_angle_to_quat(pose_aa_mj.reshape(-1, 3)).reshape(n_frames, 24, 4)

    skeleton_tree = get_skeleton_tree(gender, beta_key, asset_root)
    root_trans_offset = root_trans + skeleton_tree.local_translation[0].to(device)

    sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        pose_quat.to(device),
        root_trans_offset.to(device),
        is_local=True,
    )

    upright_inv = _UPRIGHT_QUAT_INV.to(sk_state.global_rotation.device).view(1, 1, 4)
    pose_quat_global = quat_mul(sk_state.global_rotation, upright_inv)

    sk_state = SkeletonState.from_rotation_and_root_translation(
        skeleton_tree,
        pose_quat_global,
        root_trans_offset,
        is_local=False,
    )

    pose_quat = sk_state.local_rotation
    pose_quat_global = sk_state.global_rotation
    return root_trans_offset, pose_quat, pose_quat_global


def build_phc_motion(
    humos_motion_data: dict,
    gender: str,
    beta_key: str,
    device: str,
    asset_root: str,
) -> dict:
    n_frame = humos_motion_data["trans"].shape[0]

    phc_motion = {}
    phc_motion["beta"] = humos_motion_data["betas"][0]
    phc_motion["trans_orig"] = humos_motion_data["trans"]
    phc_motion["pose_aa"] = torch.zeros(n_frame, 24, 3)
    phc_motion["pose_aa"][:, 0, :] = humos_motion_data["root_orient"]
    phc_motion["pose_aa"][:, 1:, :] = humos_motion_data["pose_body"]
    phc_motion["pose_aa"] = phc_motion["pose_aa"].reshape(n_frame, -1)

    root_trans_offset, pose_quat, pose_quat_global = calc_pose_quat(
        gender,
        beta_key,
        phc_motion["pose_aa"],
        phc_motion["trans_orig"],
        device,
        asset_root,
    )
    phc_motion["root_trans_offset"] = root_trans_offset
    phc_motion["pose_quat"] = pose_quat
    phc_motion["pose_quat_global"] = pose_quat_global

    for key in list(phc_motion.keys()):
        phc_motion[key] = phc_motion[key].to(torch.float32)

    phc_motion["offset_height"] = humos_motion_data["offset_height"]
    phc_motion["beta_key"] = beta_key
    phc_motion["gender"] = gender
    phc_motion["fps"] = 20
    return phc_motion


def pickle_to_bytes(obj: object) -> bytes:
    buf = io.BytesIO()
    pickle.dump(obj, buf, protocol=pickle.HIGHEST_PROTOCOL)
    return buf.getvalue()


def upload_pkl_with_rclone_bytes(data: bytes, remote_file: str) -> None:
    subprocess.run(
        ["rclone", "rcat", "--retries", "3", remote_file],
        input=data,
        check=True,
    )


def process_one_file(
    humos_path: str,
    asset_root: str,
    remote_dir: str,
    dry_run: bool = False,
) -> dict:
    # Avoid CPU oversubscription when multiple worker processes are active.
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    device = "cpu"
    humos_result = torch.load(humos_path, map_location=device, weights_only=False)
    motion_id = Path(humos_path).stem

    uploaded = 0
    outputs: List[str] = []

    for gender in ["male", "female"]:
        if gender not in humos_result:
            continue

        for beta_key, humos_motion_data in humos_result[gender].items():
            phc_motion = build_phc_motion(
                humos_motion_data=humos_motion_data,
                gender=gender,
                beta_key=beta_key,
                device=device,
                asset_root=asset_root,
            )
            motion_key = f"{motion_id}_{gender}_{beta_key}"
            remote_file = f"{remote_dir}/{motion_key}.pkl"
            payload = {motion_key: phc_motion}

            if not dry_run:
                upload_pkl_with_rclone_bytes(pickle_to_bytes(payload), remote_file)

            uploaded += 1
            outputs.append(motion_key)

    return {
        "file": humos_path,
        "uploaded": uploaded,
        "outputs": outputs,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert HUMOS .pt files to PHC .pkl and upload them in parallel."
    )
    parser.add_argument(
        "--input-folder",
        default=DEFAULT_INPUT_FOLDER,
        help="Root folder that contains HUMOS .pt files.",
    )
    parser.add_argument(
        "--asset-root",
        default=DEFAULT_ASSET_ROOT,
        help="ASE asset root that contains mjcf/smpl/*.xml.",
    )
    parser.add_argument(
        "--remote-dir",
        default=DEFAULT_REMOTE_DIR,
        help="Rclone remote directory, for example gdrive:humos_phc_results.",
    )
    parser.add_argument(
        "--pattern",
        default="*.pt",
        help="Glob pattern for motion files inside input-folder.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 1) // 2 or 1)),
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Convert only; do not upload with rclone.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pattern = os.path.join(args.input_folder, "**", args.pattern)
    files = sorted(glob(pattern, recursive=True))
    if not files:
        print(f"No files found under: {pattern}")
        return

    print(f"Found {len(files)} files")
    print(f"Using {args.workers} worker processes")
    print(f"Asset root: {args.asset_root}")
    print(f"Remote dir: {args.remote_dir}")

    from tqdm import tqdm

    total_outputs = 0
    failures: List[Tuple[str, str]] = []

    processed_log = os.path.join("/", "home","hlz","repos","PHC", "processed_motion_ids.txt")

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        future_to_file = {
            executor.submit(
                process_one_file,
                humos_path,
                args.asset_root,
                args.remote_dir,
                args.dry_run,
            ): humos_path
            for humos_path in files
        }

        with open(processed_log, "a", encoding="utf-8") as f_log:
            with tqdm(total=len(files), desc="humos2phc", unit="file") as pbar:
                for future in as_completed(future_to_file):
                    humos_path = future_to_file[future]
                    try:
                        result = future.result()
                        total_outputs += result["uploaded"]

                        motion_id = Path(humos_path).stem
                        f_log.write(motion_id + "\n")
                        f_log.flush()

                        pbar.set_postfix_str(
                            f"{os.path.basename(humos_path)} -> {result['uploaded']} outputs"
                        )
                    except Exception as exc:
                        failures.append((humos_path, repr(exc)))
                        pbar.set_postfix_str(f"FAILED: {os.path.basename(humos_path)}")
                    finally:
                        pbar.update(1)

    print(f"Completed files: {len(files) - len(failures)}/{len(files)}")
    print(f"Total outputs: {total_outputs}")

    if failures:
        print("Failures:")
        for humos_path, err in failures:
            print(f" - {humos_path}: {err}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
