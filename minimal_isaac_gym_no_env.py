# minimal_isaac_gym_no_env.py
# Works with Isaac Gym Preview (isaacgym 1.x). Does not create or step any envs.

from isaacgym import gymapi, gymutil
import numpy as np
import math
import os

def main():
    # Parse CLI args (e.g., --graphics_device_id, --compute_device_id, --headless)
    args = gymutil.parse_arguments(
        description="Minimum Isaac Gym example (no envs, no stepping)"
    )

    gym = gymapi.acquire_gym()

    # Create a simulator (required to validate GPU/PhysX setup), but no envs.
    sim_params = gymapi.SimParams()
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.dt = 1.0 / 60.0
    sim_params.substeps = 2
    sim_params.use_gpu_pipeline = True
    sim_params.physx.use_gpu = True
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 1
    

    # Headless by default if you pass --headless; no viewer is opened.
    sim = gym.create_sim(
        args.compute_device_id,
        args.graphics_device_id,
        gymapi.SIM_PHYSX,
        sim_params
    )
    if sim is None:
        raise RuntimeError("Failed to create Isaac Gym simulator.")


    # ground (nice reference plane)
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0,0,1)
    gym.add_ground(sim, plane_params)

    # --- asset ---
    asset_root = os.path.join("phc", "data", "assets", "mjcf")
    asset_file = "smpl_humanoid.xml"     # from your repo
    opts = gymapi.AssetOptions()
    opts.fix_base_link = True            # keep root fixed so it won't fall while we wiggle joints
    opts.disable_gravity = True          # optional extra stability for this demo
    asset = gym.load_asset(sim, asset_root, asset_file, opts)

    dof_count = gym.get_asset_dof_count(asset)

    # --- env + actor ---
    spacing = 2.0
    env = gym.create_env(sim, gymapi.Vec3(-spacing, -spacing, 0), gymapi.Vec3(spacing, spacing, spacing), 1)
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(0, 0, 1.0)     # spawn a bit above ground
    actor = gym.create_actor(env, asset, pose, "humanoid", 0, 1)

    # DOF drives: position control with modest stiffness/damping
    dof_props = gym.get_actor_dof_properties(env, actor)
    dof_props["driveMode"].fill(gymapi.DOF_MODE_POS)
    dof_props["stiffness"].fill(80.0)
    dof_props["damping"].fill(5.0)
    gym.set_actor_dof_properties(env, actor, dof_props)

    # Initial DOF state (zeros)
    dof_states = gym.get_actor_dof_states(env, actor, gymapi.STATE_ALL)
    dof_states["pos"][:] = 0.0
    dof_states["vel"][:] = 0.0
    gym.set_actor_dof_states(env, actor, dof_states, gymapi.STATE_ALL)

    # Viewer
    viewer = gym.create_viewer(sim, gymapi.CameraProperties())
    if viewer is None: raise RuntimeError("Failed to create viewer")
    cam_pose = gymapi.Transform()
    cam_pose.p = gymapi.Vec3(3.5, 3.5, 2.0)
    gym.viewer_camera_look_at(viewer, None, cam_pose.p, gymapi.Vec3(0,0,1.0))

    # --- simple motion: sinusoid on all hinge joints ---
    t = 0.0
    tgt = np.zeros(dof_count, dtype=np.float32)
    amp = 0.25             # radians
    freq = 0.6             # Hz
    phase = np.linspace(0, math.pi, dof_count, dtype=np.float32)  # small phase offsets

    while not gym.query_viewer_has_closed(viewer):
        t += sim_params.dt
        tgt[:] = amp * np.sin(2*math.pi*freq*t + phase)

        # send targets
        gym.set_actor_dof_position_targets(env, actor, tgt)

        gym.simulate(sim)
        gym.fetch_results(sim, True)
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__ == "__main__":
    main()
