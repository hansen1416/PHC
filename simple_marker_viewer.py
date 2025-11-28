"""Small standalone viewer that shows motion markers on a ground plane.

This script builds a single-environment Isaac Gym simulation, spawns one
marker actor per joint from the provided motion file, and teleports those
actors each frame using the reference positions from ``MotionLibSMPL``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from isaacgym import gymapi, gymtorch
import torch

from simple_marker_update import build_marker, build_marker_actor_ids
from simple_motion_loader import load_first_frame_rb_positions

device = (torch.device("cuda", index=0) if torch.cuda.is_available() else torch.device("cpu"))


def create_sim(gym: gymapi.Gym) -> gymapi.Sim:
    """Create a PhysX simulator with a z-up ground plane."""

    # configure sim
    sim_params = gymapi.SimParams()
    sim_params.dt = dt = 1.0 / 60.0
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6

    # sim_params.physx.num_velocity_iterations = 0
    # Use velocity iterations to avoid unstable contact resolution that can
    # throw the humanoid into the air during visualization.
    sim_params.physx.num_velocity_iterations = 1
    # Provide small offsets to keep contacts well-behaved and reduce the
    # likelihood of the character being launched from the ground.
    sim_params.physx.contact_offset = 0.02
    sim_params.physx.rest_offset = 0.0

    sim_params.physx.num_threads = 0
    sim_params.physx.use_gpu = True
    sim_params.use_gpu_pipeline = True

    compute_device = 0
    graphics_device = 0

    sim = gym.create_sim(compute_device, graphics_device, gymapi.SIM_PHYSX, sim_params)
    if sim is None:
        raise RuntimeError("Failed to create gym Sim")

    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
    gym.add_ground(sim, plane_params)

    return sim


def create_env_and_markers(
    gym: gymapi.Gym, sim: gymapi.Sim, num_markers: int
) -> Tuple[gymapi.Env, list[int]]:
    """Spawn a single env with ``num_markers`` marker actors."""

    num_envs = 1
    num_actors = 1
    num_per_row = 5
    spacing = 5
    env_lower = gymapi.Vec3(-spacing, spacing, 0)
    env_upper = gymapi.Vec3(spacing, spacing, spacing)

    env = gym.create_env(sim,  env_lower, env_upper, num_per_row)

    marker_handles: list[int] = build_marker(gym, sim, env)

    marker_actor_ids = build_marker_actor_ids(marker_handles, num_envs, num_actors, device)

    return env, marker_actor_ids


def set_camera_pose(gym: gymapi.Gym, viewer: gymapi.Viewer) -> None:
    cam_pos = gymapi.Vec3(0, -10.0, 3)
    cam_target = gymapi.Vec3(0, 0, 0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)


def random_tensor_4x3_1_2(device=None, dtype=torch.float32):
    """
    Generate a random tensor of shape (4, 3) with values in [1, 2).
    """
    t = torch.rand(24, 3, device=device, dtype=dtype)  # in [0, 1)
    return 1.0 + t  # shift to [1, 2)


def main(motion_file: str) -> None:
    gym = gymapi.acquire_gym()
    sim = create_sim(gym)
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise RuntimeError("Failed to create gym viewer")
    set_camera_pose(gym, viewer)

    # ref_rb_pos, motion_lib = load_first_frame_rb_positions(motion_file)

    # print(ref_rb_pos)
    # exit()

    num_markers = 24
    env, marker_actor_ids = create_env_and_markers(gym, sim, num_markers)

    root_state_tensor = gym.acquire_actor_root_state_tensor(sim)
    root_states = gymtorch.wrap_tensor(root_state_tensor)

    ref_rb_pos = random_tensor_4x3_1_2(device)

    flag = 0

    while not gym.query_viewer_has_closed(viewer):

        # root_states[:, 0] = root_states[:, 0]  # no-op write

        if flag == 0:

            root_states[:, 0:3].copy_(ref_rb_pos[0])

            gym.set_actor_root_state_tensor_indexed(
                sim,
                root_state_tensor,
                gymtorch.unwrap_tensor(marker_actor_ids),
                len(marker_actor_ids),
            )

            flag += 1
        
        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)


    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)


if __name__ == "__main__":
    # import argparse

    # parser = argparse.ArgumentParser(description="Preview reference markers on a ground plane")
    # parser.add_argument("motion_file", type=str, help="Path to AMASS-style motion pickle")
    # args = parser.parse_args()
    
    # main(args.motion_file)

    motion_file = "/home/hlz/datasets/AMASS/pkls/0-ACCAD_Female1Running_c3d_C4-Runtowalk1_poses.pkl"

    main(motion_file)