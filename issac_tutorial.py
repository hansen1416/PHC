from isaacgym import gymapi
import math
import random
import numpy as np
import os
import joblib

import torch
from isaacgym import gymtorch

device = "cpu"

def get_axis_params(value, axis_idx, x_value=0.0, dtype=float, n_dims=3):
    """construct arguments to `Vec` according to axis index."""
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.0
    params = np.where(zs == 1.0, value, zs)
    params[0] = x_value
    return list(params.astype(dtype))

@torch.jit.script
def torch_rand_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    return (upper - lower) * torch.rand(*shape, device=device) + lower

gym = gymapi.acquire_gym()


# get default set of parameters
sim_params = gymapi.SimParams()

# set common parameters
sim_params.dt = 1 / 60
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

# set PhysX-specific parameters
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 6
sim_params.physx.num_velocity_iterations = 1
sim_params.physx.contact_offset = 0.01
sim_params.physx.rest_offset = 0.0

# set Flex-specific parameters
sim_params.flex.solver_type = 5
sim_params.flex.num_outer_iterations = 4
sim_params.flex.num_inner_iterations = 20
sim_params.flex.relaxation = 0.8
sim_params.flex.warm_start = 0.5

sim_params.use_gpu_pipeline = True
sim_params.physx.use_gpu = True

compute_device_id = 0
graphics_device_id = 0
physics_engine = gymapi.SIM_PHYSX

# create sim with these parameters
sim = gym.create_sim(compute_device_id, graphics_device_id, physics_engine, sim_params)

gym.prepare_sim(sim)
# The state of each root body is represented using 13 floats with the same layout 
# as GymRigidBodyState: 3 floats for position, 4 floats for quaternion, 3 floats 
# for linear velocity, and 3 floats for angular velocity.
_root_tensor = gym.acquire_actor_root_state_tensor(sim)
# In order to access the contents of the tensor, 
# you can wrap it in a PyTorch Tensor object, 
# using the provided gymtorch interop module:
root_tensor = gymtorch.wrap_tensor(_root_tensor)

root_positions = root_tensor[:, 0:3]
root_orientations = root_tensor[:, 3:7]
root_linvels = root_tensor[:, 7:10]
root_angvels = root_tensor[:, 10:13]



# configure the ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1) # z-up!
plane_params.distance = 0
plane_params.static_friction = 1
plane_params.dynamic_friction = 1
plane_params.restitution = 0

# create the ground plane
gym.add_ground(sim, plane_params)

# asset_root = "./isaacgym/assets"
# asset_file = "urdf/franka_description/robots/franka_panda.urdf"
# asset = gym.load_asset(sim, asset_root, asset_file)

asset_root = os.path.join("phc", "data", "assets", "mjcf")
asset_file = "smpl_humanoid.xml"
asset = gym.load_asset(sim, asset_root, asset_file)

# set up the env grid
env_spacing = 2.0
env_lower = gymapi.Vec3(-env_spacing, 0.0, -env_spacing)
env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)

# cache some common handles for later use
env = gym.create_env(sim, env_lower, env_upper, 1)


char_h = 0.89
up_axis_idx = 2

pos = torch.tensor(get_axis_params(char_h, up_axis_idx)).to(device)
pos[:2] += torch_rand_float(-1., 1., (2, 1), device=device).squeeze(1)

start_pose = gymapi.Transform()
start_pose.p = gymapi.Vec3(*pos)
start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

actor_handle = gym.create_actor(env, asset, start_pose, "MyActor", 0, 1)

props = gym.get_actor_dof_properties(env, actor_handle)
props["driveMode"].fill(gymapi.DOF_MODE_EFFORT)
# props["driveMode"].fill(gymapi.DOF_MODE_POS)
# props["driveMode"].fill(gymapi.DOF_MODE_VEL)
props["stiffness"].fill(0.0)
props["damping"].fill(0.0)
gym.set_actor_dof_properties(env, actor_handle, props)

lower_limits = props['lower']
upper_limits = props['upper']

cam_props = gymapi.CameraProperties()
viewer = gym.create_viewer(sim, cam_props)

cam_pos = gymapi.Vec3(5.0, 0.0, 3.0)  # Camera position (x=2, y=2, z=2 or any desired values)
cam_target = gymapi.Vec3(0.0, 0.0, 0.0)  # Point for camera to look at (x=0, y=0, z=0 or any desired values)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

num_dofs = gym.get_actor_dof_count(env, actor_handle)


PHC_RESULT = os.path.join("/",
    "home", "hlz", "repos", "PHC", "output", "HumanoidIm",
    "phc_kp_mcp_iccv", "phc_act", "0-ACCAD_Male2General_c3d_A11-Crawl_poses",
    "noise_False_0.05_2025-09-17-21:24:04_np.pkl"
)

PHC_RESULT = os.path.join("/", "home", "hlz", "repos", "PHC", "output","HumanoidIm",
    "phc_kp_mcp_iccv","phc_act","amass_isaac_standing_upright_slim", "noise_False_0.05_2025-09-22-21:05:26.pkl"
)

data  = joblib.load(PHC_RESULT)
actions = data['clean_action'][0]   # (N, 69)

dof_count = len(props["driveMode"])

print(actions.shape, dof_count)

assert actions.shape[1] == dof_count, f"{actions.shape[1]} != {dof_count}"

# apply efforts (every frame)
# efforts = np.random.uniform(low=lower_limits, high=upper_limits, size=num_dofs).astype(np.float32)
# targets = np.random.uniform(low=lower_limits, high=upper_limits, size=num_dofs).astype(np.float32)
# vel_targets = np.random.uniform(-math.pi, math.pi, num_dofs).astype(np.float32)

while not gym.query_viewer_has_closed(viewer):

    for t in range(actions.shape[0]):
        targets = actions[t]   # one frame of action
        # gym.set_actor_dof_position_targets(env, actor_handle, targets)

        gym.apply_actor_dof_efforts(env, actor_handle, targets)
        # gym.set_actor_dof_position_targets(env, actor_handle, targets)
        # gym.set_actor_dof_velocity_targets(env, actor_handle, vel_targets)

        # step the physics
        gym.simulate(sim)
        gym.fetch_results(sim, True)

        # update the viewer
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)

        # Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        gym.sync_frame_time(sim)


gym.destroy_viewer(viewer)
gym.destroy_sim(sim)