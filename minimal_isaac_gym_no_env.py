# minimal_isaac_gym_no_env.py
# Works with Isaac Gym Preview (isaacgym 1.x). Does not create or step any envs.

from isaacgym import gymapi, gymutil
import numpy as np
import math
import os
import joblib

PHC_RESULT = os.path.join("/",
    "home", "hlz", "repos", "PHC", "output", "HumanoidIm",
    "phc_kp_mcp_iccv", "phc_act", "0-ACCAD_Male2General_c3d_A11-Crawl_poses",
    "noise_False_0.05_2025-09-17-21:24:04_np.pkl"
)

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
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

    

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
    # keep root fixed so it won't fall while we wiggle joints
    opts.fix_base_link = False            
    # optional extra stability for this demo
    opts.disable_gravity = False          
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
    # dof_props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
    dof_props["stiffness"].fill(4000.0)
    dof_props["damping"].fill(400.0)
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

    data  = joblib.load(PHC_RESULT)
    actions = data['clean_action'][0]   # (N, 69)

    dprops = gym.get_actor_dof_properties(env, actor)
    dof_count = len(dprops["driveMode"]) #  69

    assert actions.shape[1] == dof_count, f"{actions.shape[1]} != {dof_count}"

    # --- simple motion: sinusoid on all hinge joints ---

    while not gym.query_viewer_has_closed(viewer):

        for t in range(actions.shape[0]):
            tgt = actions[t]   # one frame of action

            gym.set_actor_dof_position_targets(env, actor, tgt)
            # gym.set_dof_position_target_tensor(env, actor, tgt)

            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.sync_frame_time(sim)
            gym.draw_viewer(viewer, sim, True)


    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)

if __name__ == "__main__":
    main()
