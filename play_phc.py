import joblib
from isaacgym import gymapi
import numpy as np

file_path = 'output/HumanoidIm/phc_kp_mcp_iccv/phc_act/0-ACCAD_Male2General_c3d_A11-Crawl_poses/noise_False_0.05_2025-09-17-21:24:04.pkl'


# Load motion data
data = joblib.load(file_path)
actions = data['env_action']  # [N, act_dim]
# Optionally: obs = data['obs']

# 1. Initialize Isaac Gym
gym = gymapi.acquire_gym()

sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.dt = 1/60.
sim_params.use_gpu_pipeline = True
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# 2. Load Asset (adjust asset path/type as needed)
asset_root = "phc/data/assets"
humanoid_asset_file = "mjcf/smpl_humanoid.xml"  # or .xml, .mjcf, etc.
humanoid_asset = gym.load_asset(sim, asset_root, humanoid_asset_file, gymapi.AssetOptions())

# 3. Create Env and Add Actor
env = gym.create_env(sim, gymapi.Vec3(-1, 0, 0), gymapi.Vec3(1, 1, 1), 1)
humanoid_handle = gym.create_actor(env, humanoid_asset, gymapi.Transform(), "humanoid", 0, 1)

# 4. Viewer setup
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "QUIT")

# 5. Main replay loop
for action in actions:
    # Apply action: depends on control mode (here we use set_dof_position_target as an example)
    gym.set_actor_dof_position_targets(env, humanoid_handle, action)
    gym.simulate(sim)
    gym.fetch_results(sim, True)
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)
    for event in gym.query_viewer_action_events(viewer):
        if event.action == "QUIT" and event.value > 0:
            exit()

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)