# Step-by-Step Guide: Basic Imitation Learning on AMASS Motion with PHC

This guide helps you get started with imitation learning using the [PHC](https://github.com/hansen1416/PHC) repository and AMASS motion data.

---

## 1. Data Preparation

- Prepare your AMASS motion data in the required format.
- Use the script: `scripts/data_process/fit_smpl_motion.py`
- Required fields in the data:
  - `pose_aa`: pose in axis-angle format
  - `gender`: subject gender
  - `trans`: root translation
  - `betas`: SMPL shape parameters
  - `fps`: motion framerate

**Example:**
```python
def load_amass_data(data_path):
    entry_data = dict(np.load(open(data_path, "rb"), allow_pickle=True))
    framerate = entry_data['mocap_framerate']
    root_trans = entry_data['trans']
    pose_aa = np.concatenate([entry_data['poses'][:, :66], np.zeros((root_trans.shape[0], 6))], axis=-1)
    betas = entry_data['betas']
    gender = entry_data['gender']
    return {
        "pose_aa": pose_aa,
        "gender": gender,
        "trans": root_trans, 
        "betas": betas,
        "fps": framerate
    }
```

---

## 2. Set Up Configuration

Choose a config for imitation learning. Example command from README:

```bash
python phc/run_hydra.py learning=im_mcp \
    exp_name=phc_kp_mcp_iccv \
    test=True \
    env=env_im_getup_mcp \
    robot=smpl_humanoid \
    robot.freeze_hand=True \
    robot.box_body=False \
    env.z_activation=relu \
    env.motion_file=sample_data/amass_isaac_standing_upright_slim.pkl \
    env.models=['output/HumanoidIm/phc_kp_pnn_iccv/Humanoid.pth'] \
    env.num_envs=1 \
    env.obs_v=7 \
    headless=False \
    epoch=-1
```

- Replace `env.motion_file` with your processed AMASS motion file path.

---

## 3. Key Components in PHC

a. **HumanoidIm Class**  File: `phc/env/tasks/humanoid_im.py`  

- Handles loading and processing of motion data
- Manages the imitation environment
- Computes observations and rewards

b. **IMAmpAgent Class**  File: `phc/learning/im_amp.py`  

- Implements the imitation learning agent
- Handles training and evaluation
- Tracks metrics like MPJPE (Mean Per Joint Position Error)

---

## 4. Training Process

1. **Load Motion Data**  
   Example config (see code):
   ```python
   motion_lib_cfg = EasyDict({
       "motion_file": motion_train_file,
       "device": torch.device("cpu"),
       "fix_height": FixHeightMode.full_fix,
       "min_length": self._min_motion_len,
       "max_length": -1,
       "smpl_type": self.humanoid_type,
       "randomrize_heading": True,
       "step_dt": self.dt,
   })
   ```

2. The agent will:
- Sample motions from the motion library
- Generate observations of the current state
- Compute rewards based on how well the agent imitates the reference motion
- Update the policy to improve imitation performance

---

## 5. Monitoring Progress

The system provides several metrics to track progress:

- Success rate for motion imitation
- MPJPE (Mean Per Joint Position Error)
- Body position predictions vs ground truth comparisons

---

## 6. Getting Started Steps

## 7.First, prepare your AMASS motion data and convert it to the required format

## 8.Place your processed motion file in the sample_data directory

## 9.Modify the training command above, replacing env.motion_file with your motion file path

## 10.Run the training command to start imitation learning

## 11.Monitor the training progress through the provided metrics

For a basic test run, you can use the example command provided in the README, which uses a pre-processed AMASS motion file (amass_isaac_standing_upright_slim.pkl).

