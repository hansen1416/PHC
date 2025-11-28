"""Tiny helper for grabbing reference body positions from a motion file.

This module shows the bare-minimum sequence of calls needed to pull the
first-frame rigid-body positions (``rg_pos``) from an AMASS-style motion
pickle by driving ``MotionLibSMPL`` directly.
"""
from __future__ import annotations

from typing import Tuple

import joblib
import numpy as np
import torch
from easydict import EasyDict
from poselib.poselib.skeleton.skeleton3d import SkeletonTree

from phc.utils.motion_lib_base import FixHeightMode
from phc.utils.motion_lib_smpl import MotionLibSMPL


def _build_motion_lib_cfg(motion_file: str, device: torch.device) -> EasyDict:
    """Return a minimal config object for ``MotionLibSMPL``.

    The library expects a handful of attributes on its config; here we provide
    the defaults necessary to load a single clip without multiprocessing or
    heading randomization.
    """

    return EasyDict(
        motion_file=motion_file,
        device=device,
        fix_height=FixHeightMode.no_fix,
        min_length=-1,
        max_length=-1,
        im_eval=True,
        multi_thread=False,
        smpl_type="smpl",
        randomrize_heading=False,
        step_dt=1 / 30.0,
    )


def load_first_frame_rb_positions(
    motion_file: str, device: torch.device | None = None
) -> Tuple[torch.Tensor, MotionLibSMPL]:
    """Load ``ref_rb_pos`` (``rg_pos``) for the first frame of ``motion_file``.

    Args:
        motion_file: Path to the AMASS-format motion pickle.
        device: Torch device to place the loaded tensors on. Defaults to CUDA
            if available, otherwise CPU.

    Returns:
        A tuple ``(ref_rb_pos, motion_lib)`` where ``ref_rb_pos`` is shaped
        ``(1, num_bodies, 3)`` and contains the world-space joint positions for
        frame 0, and ``motion_lib`` is the initialized ``MotionLibSMPL``
        instance in case more frames are needed.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Grab the raw dict entry from the motion pickle so we can recover the
    # skeleton tree and shape parameters expected by MotionLibSMPL.
    motion_data = joblib.load(motion_file)
    first_key = next(iter(motion_data))
    motion_entry = motion_data[first_key]

    skeleton_dict = motion_entry.get("skeleton_tree")
    if skeleton_dict is None:
        raise ValueError("motion file is missing a serialized skeleton_tree")
    skeleton_tree = SkeletonTree.from_dict(skeleton_dict)

    gender_beta = motion_entry.get("gender_beta", np.zeros(17, dtype=np.float32))
    gender_beta = torch.as_tensor(gender_beta, dtype=torch.float32)

    # Limb weights are required by the loader but not used for position queries;
    # a zero vector of reasonable length is sufficient here.
    limb_weights = [np.zeros(10, dtype=np.float32)]

    motion_lib_cfg = _build_motion_lib_cfg(motion_file, device)
    motion_lib = MotionLibSMPL(motion_lib_cfg)
    motion_lib.load_motions(
        skeleton_trees=[skeleton_tree],
        gender_betas=[gender_beta],
        limb_weights=limb_weights,
        random_sample=False,
        max_len=-1,
    )

    # Pull frame zero of motion zero. ``rg_pos`` holds the rigid-body positions
    # ordered the same way they appear in ``skeleton_tree``.
    motion_ids = torch.tensor([0], device=device, dtype=torch.long)
    motion_times = torch.tensor([0], device=device, dtype=torch.long)
    motion_state = motion_lib.get_motion_state(motion_ids, motion_times)
    ref_rb_pos = motion_state["rg_pos"].to(device)

    return ref_rb_pos, motion_lib


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dump first-frame ref positions")
    parser.add_argument("motion_file", type=str, help="Path to motion pickle")
    args = parser.parse_args()

    ref_rb_pos, _ = load_first_frame_rb_positions(args.motion_file)
    print(ref_rb_pos)