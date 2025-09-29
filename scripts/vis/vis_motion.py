"""
Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.

Visualize motion library


Module: vis_motion.py
Description: Visualizes SMPL-based human motions using Isaac Gym. 
oads an SMPL robot model, simulates physics,
and animates motions from a library. Supports keyboard controls for motion switching and debugging.
Dependencies: isaacgym, torch, numpy, joblib, etc.
Usage: Run directly to start the simulation viewer.

"""
import glob
import os
import sys
import pdb
import os.path as osp

sys.path.append(os.getcwd())

import joblib
import numpy as np
from isaacgym import gymapi, gymutil, gymtorch
import torch
from phc.utils.motion_lib_smpl import MotionLibSMPL as MotionLibSMPL
from smpl_sim.smpllib.smpl_local_robot import SMPL_Robot
from poselib.poselib.skeleton.skeleton3d import SkeletonTree
from phc.utils.flags import flags
from easydict import EasyDict
from phc.utils.motion_lib_base import FixHeightMode



def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)

def action_to_pd_target(action, device='cuda:0'):

    action_offset = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       device=device)

    action_scale = torch.tensor([3.1416, 3.1416, 3.1416, 3.1416, 5.0000, 3.1416, 3.1416, 3.1416, 3.1416,
            3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 5.0000, 3.1416,
            3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416,
            3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416,
            3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416,
            3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416,
            3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416,
            3.1416, 3.1416, 3.1416, 3.1416, 3.1416, 3.1416], device=device)

    # _freeze_hand = True

    pd_target = action_offset + action_scale * action

    SMPL_MUJOCO_NAMES = ['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 
                     'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']

    _dof_names = SMPL_MUJOCO_NAMES[1:]  # exclude pelvis

    pd_target[_dof_names.index("L_Hand") * 3:(_dof_names.index("L_Hand") * 3 + 3)] = 0
    pd_target[_dof_names.index("R_Hand") * 3:(_dof_names.index("R_Hand") * 3 + 3)] = 0

    return pd_target

# parse arguments
args = gymutil.parse_arguments(description="Joint monkey: Animate degree-of-freedom ranges",
                               custom_parameters=[
                                   {
                                    "name": "pos_index",     # no '--', since it's positional
                                    "type": int,
                                    "default": 0,
                                    "help": "Index of something",
                                    "positional": True       # makes it positional
                                },
                                   {
                                   "name": "--speed_scale",
                                   "type": float,
                                   "default": 1.0,
                                   "help": "Animation speed scale"
                               }, {
                                   "name": "--show_axis",
                                   "action": "store_true",
                                   "help": "Visualize DOF axis"
                               }])


# motion_file = "data/amass/pkls/0-ACCAD_Female1General_c3d_A3-Swing_poses.pkl"
# motion_file = "data/amass/pkls/0-ACCAD_Female1Running_c3d_C4-Runtowalk1_poses.pkl"

results_pair = [
    {"motion_file": "data/amass/pkls/0-ACCAD_Male2General_c3d_A11-Crawl_poses.pkl",
     "phc_result": os.path.join("/",
        "home", "hlz", "repos", "PHC", "output", "HumanoidIm",
        "phc_kp_mcp_iccv", "phc_act", "0-ACCAD_Male2General_c3d_A11-Crawl_poses",
        "noise_False_0.05_2025-09-28-22:19:46.pkl")},
    {"motion_file": "sample_data/amass_isaac_standing_upright_slim.pkl",
        "phc_result": os.path.join("/",
            "home", "hlz", "repos", "PHC", "output", "HumanoidIm",
            "phc_kp_mcp_iccv", "phc_act", "amass_isaac_standing_upright_slim",
            "noise_False_0.05_2025-09-29-16:02:08.pkl")},
    {"motion_file": "data/amass/pkls/0-ACCAD_MartialArtsWalksTurns_c3d_E15-blockleftmiddle_poses.pkl",
        "phc_result": os.path.join("/",
            "home", "hlz", "repos", "PHC", "output", "HumanoidIm",
            "phc_kp_mcp_iccv", "phc_act", "0-ACCAD_MartialArtsWalksTurns_c3d_E15-blockleftmiddle_poses",
            "noise_False_0.05_2025-09-29-16:18:10.pkl")},
    {
    "motion_file": "data/amass/pkls/0-ACCAD_Female1Running_c3d_C4-Runtowalk1_poses.pkl",
    "phc_result": "output/HumanoidIm/phc_kp_mcp_iccv/phc_act/0-ACCAD_Female1Running_c3d_C4-Runtowalk1_poses/noise_False_0.05_2025-09-29-21:27:25.pkl"
    },
]

result_i = int(args.pos_index)

motion_file = results_pair[result_i]["motion_file"]
phc_result = results_pair[result_i]["phc_result"]


# initialize gym
gym = gymapi.acquire_gym()

# configure sim
sim_params = gymapi.SimParams()
sim_params.dt = dt = 1.0 / 60.0
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
if args.physics_engine == gymapi.SIM_FLEX:
    pass
elif args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 6
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline

if not args.use_gpu_pipeline:
    print("WARNING: Forcing CPU pipeline.")

sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    print("*** Failed to create sim")
    quit()

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    print("*** Failed to create viewer")
    quit()

asset_root = os.path.join("phc", "data", "assets", "mjcf")
asset_file = "smpl_humanoid.xml"

sk_tree = SkeletonTree.from_mjcf(osp.join(asset_root, asset_file))

asset_options = gymapi.AssetOptions()

print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
asset = gym.load_asset(sim, asset_root, asset_file, asset_options)

# set up the env grid
num_envs = 1
num_per_row = 5
spacing = 5
env_lower = gymapi.Vec3(-spacing, spacing, 0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)

# position the camera
cam_pos = gymapi.Vec3(0, -5.0, 3)
cam_target = gymapi.Vec3(0, 0, 0)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

# cache useful handles
envs = []
actor_handles = []

num_dofs = gym.get_asset_dof_count(asset)

# create env
env = gym.create_env(sim, env_lower, env_upper, num_per_row)
envs.append(env)

# add actor
pose = gymapi.Transform()
pose.p = gymapi.Vec3(-0.2019,  0.0334,  0.8900)
pose.r = gymapi.Quat(0, 0.0, 0.0, 1)

actor_handle = gym.create_actor(env, asset, pose, "actor", 0, 1)
actor_handles.append(actor_handle)

# set default DOF positions
dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)
gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_ALL)

props = gym.get_actor_dof_properties(env, actor_handle)
props["driveMode"].fill(gymapi.DOF_MODE_POS)            # PD position mode
# Reasonable generic gains (tune to match training if needed)

gym.set_actor_dof_properties(env, actor_handle, props) 

# Setup Motion
body_ids = []
key_body_names = ["R_Ankle", "L_Ankle", "R_Wrist", "L_Wrist"]
for body_name in key_body_names:
    body_id = gym.find_actor_rigid_body_handle(envs[0], actor_handles[0], body_name)
    assert (body_id != -1)
    body_ids.append(body_id)
gym.prepare_sim(sim)
body_ids = np.array(body_ids)



motion_data = joblib.load(motion_file)

device = (torch.device("cuda", index=0) if torch.cuda.is_available() else torch.device("cpu"))

motion_lib_cfg = EasyDict({
                "motion_file": motion_file,
                "device": torch.device("cpu"),
                "fix_height": FixHeightMode.no_fix,
                "min_length": -1,
                "max_length": -1,
                "im_eval": False,
                "multi_thread": False ,
                "smpl_type": "smpl",
                "randomrize_heading": True,
                "device": device,
                "step_dt": 1/60,
            })

motion_lib = MotionLibSMPL(motion_lib_cfg)

motion_lib.load_motions(skeleton_trees=[sk_tree], gender_betas=[torch.zeros(17)], limb_weights=[np.zeros(10)], random_sample=False)

rigidbody_state = gym.acquire_rigid_body_state_tensor(sim)
rigidbody_state = gymtorch.wrap_tensor(rigidbody_state)
rigidbody_state = rigidbody_state.reshape(num_envs, -1, 13)

actor_root_state = gym.acquire_actor_root_state_tensor(sim)
actor_root_state = gymtorch.wrap_tensor(actor_root_state)

motion_id = 0
time_step = 0

# tensor([0], device='cuda:0', dtype=torch.int32)
env_ids = torch.arange(num_envs).int().to(args.sim_device)

motion_len = motion_lib.get_motion_length(motion_id).item()


motion_time = time_step % motion_len
# motion_time = 0

motion_res = motion_lib.get_motion_state(torch.tensor([motion_id]).to(args.compute_device_id), torch.tensor([motion_time]).to(args.compute_device_id))

root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, smpl_params, limb_weights, pose_aa, rb_pos, rb_rot, body_vel, body_ang_vel = \
            motion_res["root_pos"], motion_res["root_rot"], motion_res["dof_pos"], motion_res["root_vel"], motion_res["root_ang_vel"], motion_res["dof_vel"], \
            motion_res["motion_bodies"], motion_res["motion_limb_weights"], motion_res["motion_aa"], motion_res["rg_pos"], motion_res["rb_rot"], motion_res["body_vel"], motion_res["body_ang_vel"]


root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1).repeat(num_envs, 1)

dof_state = torch.stack([dof_pos, torch.zeros_like(dof_pos)], dim=-1).squeeze().repeat(num_envs, 1)



# ------- load motion action results -------
data  = joblib.load(phc_result)
actions = data['env_action'][0]   # (N, 69)

actions = torch.tensor(actions, dtype=torch.float32).to(device)

# print(actions.shape)

t_idx = 0
tota_steps = actions.shape[0]

# Pre-allocate a device tensor for per-step targets
pd_target = torch.empty_like(actions[0])  # shape (A,)

# print(pd_target.shape)

while not gym.query_viewer_has_closed(viewer):

    if t_idx == 0:
        gym.set_actor_root_state_tensor_indexed(sim, gymtorch.unwrap_tensor(root_states), gymtorch.unwrap_tensor(env_ids), len(env_ids))
        gym.set_dof_state_tensor_indexed(sim, gymtorch.unwrap_tensor(dof_state), gymtorch.unwrap_tensor(env_ids), len(env_ids))

        gym.simulate(sim)
        gym.fetch_results(sim, True)

        gym.refresh_actor_root_state_tensor(sim)
        gym.refresh_rigid_body_state_tensor(sim)

    # step the physics

    # gym.simulate(sim)

    pd_tar = action_to_pd_target(actions[t_idx % tota_steps], device=device)

    pd_target[:] = pd_tar

    # If actions already match DOF ordering and count:
    # (If you needed expansion from a reduced action set, use the expanded vector instead.)
    pd_tar_tensor = gymtorch.unwrap_tensor(pd_target)

     # set PD position targets (this produces torques via Kp, Kd)
    gym.set_dof_position_target_tensor(sim, pd_tar_tensor)  # Isaac PD path.
    
    gym.simulate(sim)
    
    gym.fetch_results(sim, True)

    # update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)


    # gym.refresh_dof_state_tensor(sim)
    # gym.refresh_actor_root_state_tensor(sim)
    # gym.refresh_rigid_body_state_tensor(sim)

    # gym.refresh_force_sensor_tensor(sim)
    # gym.refresh_dof_force_tensor(sim)
    # gym.refresh_net_contact_force_tensor(sim)

    # Wait for dt to elapse in real time.
    # This synchronizes the physics simulation with the rendering rate.
    gym.sync_frame_time(sim)
    
    # time_step += dt
    t_idx += 1

    if t_idx >= tota_steps:
        t_idx = 0


print("Done")

gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
