import joblib
import numpy as np
import re


file_path = 'output/HumanoidIm/phc_kp_mcp_iccv/phc_act/0-ACCAD_Male2General_c3d_A11-Crawl_poses/noise_False_0.05_2025-09-17-21:24:04.pkl'

def read_phc_result(file_path):
    """
    Load a PHC (Perpetual Humanoid Control) evaluation/result file.

    This helper reads a result produced by running:
      `python phc/run_hydra.py ... collect_dataset=True`
    and returns the saved Python object (usually a dict) that contains
    time-series arrays, normalization stats, and the full experiment config.

    Parameters
    ----------
    file_path : str or os.PathLike
        Path to the PHC result file saved with `joblib.dump(...)`.

    Returns
    -------
    dict
        A mapping with (at least) the following keys. Shapes refer to each
        trajectory in the lists (one per environment when recorded):

        - 'obs' : list[numpy.ndarray]
            Normalized observations over time. Each array has shape (T, D),
            dtype float32, where:
              * T = number of recorded steps in the episode.
              * D = observation dimension (equals config['env']['numObservations']).
        - 'clean_action' : list[numpy.ndarray]
            Policy actions before any environment-side filtering/noise.
            Each array has shape (T, A), dtype float32, where A is the action
            dimension (for SMPL humanoid PD control this is typically 69 = 23×3,
            but may differ by robot/control settings).
        - 'env_action' : list[numpy.ndarray]
            Actions actually applied in the environment. Same shape/dtype as
            'clean_action'. In many eval runs these are identical (no filter/noise).
        - 'key_names' : numpy.ndarray[str]
            Filename of the motion clip split into single-character strings
            (join with ''.join(...) to reconstruct a readable name).
        - 'motion_lengths' : list[int]
            Source motion length(s) in frames for each referenced clip.
        - 'reset' : numpy.ndarray[int]
            Per-step reset flags with shape (T,). A terminal step is often
            indicated by a trailing 1.
        - 'running_mean' : collections.OrderedDict
            Feature-wise running statistics used to (de)normalize observations:
              * 'running_mean' : torch.Tensor of shape (D,)
              * 'running_var'  : torch.Tensor of shape (D,)
              * 'count'        : torch.Tensor scalar
            Use these to de-normalize: x_real = obs * sqrt(var) + mean (per feature).
        - 'config' : dict
            Full experiment configuration (env, robot, learning, sim, etc.).

    Notes
    -----
    - Shapes (T, D) and (T, A) can vary with configuration:
      D == config['env']['numObservations'], A depends on the robot and control mode.
    - Values in 'obs' are normalized; use 'running_mean' / 'running_var' to recover
      physical units for analysis and plotting.

    Examples
    --------
    >>> data = read_phc_result("output/my_eval/joblib_result.pkl")
    >>> obs = data['obs'][0]             # (T, D) float32
    >>> acts = data['env_action'][0]     # (T, A) float32
    >>> import numpy as np, torch
    >>> mean = data['running_mean']['running_mean'].cpu().numpy()
    >>> var  = data['running_mean']['running_var'].cpu().numpy()
    >>> obs_denorm = obs * np.sqrt(var) + mean

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    Exception
        Any error propagated by `joblib.load` (e.g., unpickling issues).
    """
    data = joblib.load(file_path)

    # convert all torch tensors to numpy arrays for easier downstream use
    for k, v in data.items():
        if isinstance(v, list):
            # list of arrays/tensors (e.g. 'obs', 'clean_action', 'env_action')
            data[k] = [x.cpu().numpy() if hasattr(x, 'cpu') else x for x in v]
        elif isinstance(v, dict):
            # nested dict (e.g. 'running_mean')
            for k2, v2 in v.items():
                if hasattr(v2, 'cpu'):
                    data[k][k2] = v2.cpu().numpy()
        elif hasattr(v, 'cpu'):
            # single tensor (e.g. could be a single tensor in future versions)
            data[k] = v.cpu().numpy()
        # else: assume already numpy or primitive type


    # save `data` to a new file
    joblib.dump(data, file_path.replace('.pkl', '_np.pkl'), compress=True)

    return data


res = read_phc_result(file_path)

print(res.keys())

for k,v in res.items():
    # display the type and shape if applicable
    if isinstance(v, list):
        print(f"{k}: list of {len(v)} arrays")
        for i, arr in enumerate(v):
            if hasattr(arr, 'shape'):
                print(f"  [{i}] shape: {arr.shape}, dtype: {arr.dtype}")
            else:
                print(f"  [{i}] type: {type(arr)}")
    elif isinstance(v, dict):
        print(f"{k}: dict with keys {list(v.keys())}")
        for k2, v2 in v.items():
            if hasattr(v2, 'shape'):
                print(f"  {k2}: shape: {v2.shape}, dtype: {v2.dtype}")
            else:
                print(f"  {k2}: type: {type(v2)}")
    elif isinstance(v, np.ndarray):
        print(f"{k}: ndarray shape: {v.shape}, dtype: {v.dtype}")
    else:
        print(f"{k}: type: {type(v)}, value: {v}")



# data = joblib.load('data/amass/amass_train_take6.pkl')
# data = joblib.load(file_path)
# data is a dictionary where each key is a sequence, and each value is the `new_motion_out` dictionary as described above

# Example: print keys and access the first sequence
# print(data.keys())
# dict_keys(['obs', 'clean_action', 'env_action', 'key_names', 'motion_lengths', 'reset', 'running_mean', 'config'])

# for k, v in data.items():
#     # save each one as a separate file, named by the key filtered to be a valid filename
#     print(k)
#     print(v)


# sprint1 = {"sprint1": data['0-ACCAD_s009_Sprint1_poses']}

# print(sprint1.keys())

# print(sprint1['sprint1']['pose_quat_global'].shape)

# # first_seq = next(iter(data.values()))
# # print(first_seq['pose_quat_global'].shape)
# joblib.dump(sprint1, "data/amass/amass_sprint1.pkl", compress=True)


# # data = joblib.load('sample_data/amass_isaac_standing_upright_slim.pkl')

# # print(data.keys())

# # print(data['standing'].keys())