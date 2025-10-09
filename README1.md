isaacgym/python/isaacgym/torch_utils.py
line 139, def get_axis_params(value, axis_idx, x_value=0.0, dtype=float, n_dims=3):


conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -r requirement.txt

pip uninstall chumpy

pip install -e ../chumpy/

isaacgym/python$ pip install -e .


```
python phc/run_hydra.py learning=im_mcp     exp_name=phc_kp_mcp_iccv     test=True     env=env_im_getup_mcp     robot=smpl_humanoid     robot.freeze_hand=True     robot.box_body=False     env.z_activation=relu     env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']     env.num_envs=1     env.obs_v=7     headless=False     epoch=-1   im_eval=True collect_dataset=True
```

```
python phc/run_hydra.py learning=im_mcp     exp_name=phc_kp_mcp_iccv     test=True     env=env_im_getup_mcp     robot=smpl_humanoid     robot.freeze_hand=True     robot.box_body=False     env.z_activation=relu     env.motion_file=data/amass/pkls/0-ACCAD_Male2General_c3d_A11-Crawl_poses.pkl env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']     env.num_envs=1     env.obs_v=7     headless=False     epoch=-1   im_eval=True collect_dataset=True
```

```
python phc/run_hydra.py learning=im_mcp     exp_name=phc_kp_mcp_iccv     test=True     env=env_im_getup_mcp     robot=smpl_humanoid     robot.freeze_hand=True     robot.box_body=False     env.z_activation=relu     env.motion_file=data/amass/pkls/0-ACCAD_MartialArtsWalksTurns_c3d_E15-blockleftmiddle_poses.pkl env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']     env.num_envs=1     env.obs_v=7     headless=False     epoch=-1   im_eval=True collect_dataset=True
```

```
python phc/run_hydra.py learning=im_mcp     exp_name=phc_kp_mcp_iccv     test=True     env=env_im_getup_mcp     robot=smpl_humanoid     robot.freeze_hand=True     robot.box_body=False     env.z_activation=relu     env.motion_file=data/amass/pkls/0-ACCAD_Female1Running_c3d_C4-Runtowalk1_poses.pkl env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']     env.num_envs=1     env.obs_v=7     headless=False     epoch=-1   im_eval=True collect_dataset=True
```

```
python phc/run_hydra.py learning=im_mcp     exp_name=phc_kp_mcp_iccv     test=True     env=env_im_getup_mcp     robot=smpl_humanoid     robot.freeze_hand=True     robot.box_body=False     env.z_activation=relu     env.motion_file=data/amass/pkls/0-ACCAD_Female1General_c3d_A3-Swing_poses.pkl env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth']     env.num_envs=1     env.obs_v=7     headless=False     epoch=-1   im_eval=True collect_dataset=True
```

single imitation policy

Lightweight baseline:

exp_name only labels outputs and logs?

```
python phc/run_hydra.py learning=im \
exp_name=phc_prim_iccv env=env_im robot=smpl_humanoid_shape \
env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl
```

Bigger single-primitive model (more capacity, slower):
```
python phc/run_hydra.py learning=im_big exp_name=phc_prim env=env_im robot=smpl_humanoid env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl  
```