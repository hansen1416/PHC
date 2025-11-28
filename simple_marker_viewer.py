"""Small standalone viewer that shows motion markers on a ground plane.

This script builds a single-environment Isaac Gym simulation, spawns one
marker actor per joint from the provided motion file, and teleports those
actors each frame using the reference positions from ``MotionLibSMPL``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from isaacgym import gymapi
import torch

from simple_marker_update import build_marker, collect_marker_actor_ids, set_marker_positions
from simple_motion_loader import load_first_frame_rb_positions


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

    env = gym.create_env(sim, gymapi.Vec3(-2.0, -2.0, 0.0), gymapi.Vec3(2.0, 2.0, 2.0), 1)

    marker_handles: list[int] = build_marker(gym, sim, env)
    
    return env, marker_handles



def set_camera_pose(gym: gymapi.Gym, viewer: gymapi.Viewer) -> None:
    cam_pos = gymapi.Vec3(0, -10.0, 3)
    cam_target = gymapi.Vec3(0, 0, 0)
    gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)


def main(motion_file: str) -> None:
    gym = gymapi.acquire_gym()
    sim = create_sim(gym)
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None:
        raise RuntimeError("Failed to create gym viewer")
    set_camera_pose(gym, viewer)

    ref_rb_pos, motion_lib = load_first_frame_rb_positions(motion_file)
    device = ref_rb_pos.device

    num_markers = ref_rb_pos.shape[1]
    env, marker_handles = create_env_and_markers(gym, sim, num_markers)
    # marker_actor_ids = collect_marker_actor_ids(
    #     gym, [env] * num_markers, [marker_handles], device=device
    # )

    # motion_ids = torch.tensor([0], device=device, dtype=torch.long)
    # motion_len = motion_lib.get_motion_length(motion_ids)[0].item()
    # motion_dt = motion_lib._motion_dt[motion_ids][0].item()
    # motion_time = 0.0

    while not gym.query_viewer_has_closed(viewer):
        
        print(ref_rb_pos)
        print(ref_rb_pos.shape)
        print("==================================")
        
        # motion_state = motion_lib.get_motion_state(
        #     motion_ids, torch.tensor([motion_time])
        # )
        # ref_positions = motion_state["rg_pos"][0]
        # set_marker_positions(gym, sim, marker_actor_ids, ref_positions)


        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)

        # gym.simulate(sim)
        # gym.fetch_results(sim, True)
        # # update the viewer
        # gym.step_graphics(sim)
        # gym.draw_viewer(viewer, sim, True)

        # motion_time += motion_dt
        # if motion_time > motion_len:
        #     motion_time = 0.0

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