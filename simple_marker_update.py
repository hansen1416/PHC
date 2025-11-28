"""Minimal helpers for updating marker actors from reference motion positions.

The routines below assume you already built marker actors in each environment and
that `ref_rb_pos` contains the reference world-space root positions for each
marker in the same order as `marker_handles`.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import torch
from isaacgym import gymapi, gymtorch


def load_marker_asset(gym: gymapi.Gym, sim: gymapi.Sim):
    """
    load red ball marker and stores them for later instantiation
    """
    asset_root = "/home/hlz/repos/PHC/phc/data/assets/urdf/"

    asset_options = gymapi.AssetOptions()
    asset_options.angular_damping = 0.0
    asset_options.linear_damping = 0.0
    asset_options.max_angular_velocity = 0.0
    asset_options.density = 0
    asset_options.fix_base_link = True
    asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

    marker_asset = gym.load_asset(sim, asset_root, "traj_marker.urdf", asset_options)
    
    marker_asset_small = gym.load_asset(sim, asset_root, "traj_marker_small.urdf", asset_options)

    return marker_asset, marker_asset_small

def build_marker(gym: gymapi.Gym, sim: gymapi.Sim, env_ptr):

    marker_asset, _ = load_marker_asset(gym, sim)

    _marker_handles = []

    _num_joints = 24

    default_pose = gymapi.Transform()
    for i in range(_num_joints):
        # Giving hands smaller balls to indicate positions

        marker_handle = gym.create_actor(env_ptr, marker_asset, default_pose, f"marker_{i}", i, 1, 0)
        
        gym.set_rigid_body_color(env_ptr, marker_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.8, 0.0, 0.0))
   
        _marker_handles.append(marker_handle)

    return [_marker_handles]


def collect_marker_actor_ids(
    gym: gymapi.Gym,
    envs: Sequence[gymapi.Env],
    marker_handles: Iterable[int],
    device: torch.device | str = "cuda",
) -> torch.Tensor:
    """Return a tensor of actor IDs for the provided marker handles.

    Isaac Gym addresses actor root states by a flat actor index. This helper
    converts per-environment actor handles into those global indices so they can
    be used with ``set_actor_root_state_tensor_indexed``.
    """

    marker_actor_ids = []
    for env, handle in zip(envs, marker_handles):
        actor_id = gym.get_actor_index(env, handle, gymapi.DOMAIN_SIM)
        marker_actor_ids.append(actor_id)

    return torch.tensor(marker_actor_ids, dtype=torch.int32, device=device)


def set_marker_positions(
    gym: gymapi.Gym,
    sim: gymapi.Sim,
    marker_actor_ids: torch.Tensor,
    ref_rb_pos: torch.Tensor,
) -> None:
    """Write reference positions into the marker root states.

    Args:
        gym: Isaac Gym interface.
        sim: Simulator used to obtain the root-state tensor.
        marker_actor_ids: Global actor indices as returned by
            :func:`collect_marker_actor_ids`.
        ref_rb_pos: Tensor shaped ``(num_markers, 3)`` holding world-space
            positions for each marker.
    """

    # Acquire and wrap the full actor root-state tensor.
    root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
    root_states = gymtorch.wrap_tensor(root_state_tensor)

    # Slice out just the marker rows and update positions; zero the rest for a
    # clean teleport. Root-state layout is [pos(3), quat(4), lin_vel(3), ang_vel(3)].
    marker_states = root_states[marker_actor_ids]
    marker_states[:, 0:3] = ref_rb_pos
    marker_states[:, 3:7] = torch.tensor([0.0, 0.0, 0.0, 1.0], device=ref_rb_pos.device)
    marker_states[:, 7:10] = 0.0
    marker_states[:, 10:13] = 0.0

    # Push the updated rows back to the simulator.
    gym.set_actor_root_state_tensor_indexed(
        sim,
        root_state_tensor,
        marker_actor_ids,
        marker_actor_ids.numel(),
    )





