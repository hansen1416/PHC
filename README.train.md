Single primitive training:

`
python phc/run_hydra.py \
learning=im_big \
exp_name=phc_prim \
env=env_im \
robot=smpl_humanoid \
robot.freeze_hand=True \
robot.box_body=False \
env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
headless=False 
`


------------------


MCP Testing Martial Arts block left middle:

`
python phc/run_hydra.py \
  learning=im_mcp \
  exp_name=phc_kp_mcp_iccv \
  test=True \
  env=env_im_getup_mcp \
  robot=smpl_humanoid \
  robot.freeze_hand=True \
  robot.box_body=False \
  env.z_activation=relu \
  env.motion_file=/home/hlz/datasets/amass-pkls/0-ACCAD_MartialArtsWalksTurns_c3d_E15-blockleftmiddle_poses.pkl \
  "env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']" \
  env.num_envs=1 \
  env.obs_v=7 \
  headless=False \
  epoch=-1 \
  im_eval=True \
  collect_dataset=True
`

`learning=im_mcp`
“imitation + MCP-style composition”. MCP refers to a multiplicative compositional policy that combines primitives via a composer. 

`robot.freeze_hand=True`
Freezes (disables actuation or locks) hand-related joints/DoFs to simplify control and improve stability.

`im_eval=True`
Enables imitation-evaluation mode (often computing imitation metrics / using evaluation rollout settings rather than training settings).

`collect_dataset=True`
Collects and saves rollouts (states/actions/targets, etc.) into a dataset for later training/analysis.


------

MCP Testing standing_upright

python phc/run_hydra.py learning=im_mcp     exp_name=phc_kp_mcp_iccv     test=True     env=env_im_getup_mcp     robot=smpl_humanoid     robot.freeze_hand=True     robot.box_body=False     env.z_activation=relu     env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']     env.num_envs=1     env.obs_v=7     headless=False     epoch=-1   im_eval=True collect_dataset=True

------

MCP Testing standing swing

python phc/run_hydra.py learning=im_mcp     exp_name=phc_kp_mcp_iccv     test=True     env=env_im_getup_mcp     robot=smpl_humanoid     robot.freeze_hand=True     robot.box_body=False     env.z_activation=relu     env.motion_file=/home/hlz/datasets/amass-pkls/0-ACCAD_Female1General_c3d_A3-Swing_poses.pkl env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']     env.num_envs=1     env.obs_v=7     headless=False     epoch=-1   im_eval=True collect_dataset=True


------

MCP Testing run to walk (only work when stateInit == 'start')

python phc/run_hydra.py learning=im_mcp     exp_name=phc_kp_mcp_iccv     test=True     env=env_im_getup_mcp     robot=smpl_humanoid     robot.freeze_hand=True     robot.box_body=False     env.z_activation=relu     env.motion_file=/home/hlz/datasets/amass-pkls/0-ACCAD_Female1Running_c3d_C4-Runtowalk1_poses.pkl env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']     env.num_envs=1     env.obs_v=7     headless=False     epoch=-1   im_eval=True collect_dataset=True

MCP Testing run to walk (only work when stateInit == 'random')

python phc/run_hydra.py learning=im_mcp     exp_name=phc_kp_mcp_iccv     test=True     env=env_im_getup_mcp     robot=smpl_humanoid     robot.freeze_hand=True     robot.box_body=False     env.z_activation=relu     env.motion_file=/home/hlz/datasets/amass-pkls/0-ACCAD_Male2General_c3d_A11-Crawl_poses.pkl env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']     env.num_envs=1     env.obs_v=7     headless=False     epoch=-1   im_eval=True collect_dataset=True