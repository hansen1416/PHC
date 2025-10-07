import joblib
import re


# data = joblib.load('data/amass/amass_train_take6.pkl')
data = joblib.load('data/amass/amass_train_take6_upright.pkl')
# data is a dictionary where each key is a sequence, and each value is the `new_motion_out` dictionary as described above

# Example: print keys and access the first sequence
# print(data.keys())

for k, v in data.items():
    # save each one as a separate file, named by the key filtered to be a valid filename
    filename = re.sub(r'[/\s]+', '', k) + '.pkl'
    #k.replace(r'[/\s]+', '_') + '.pkl'
    joblib.dump({k: v}, f'data/amass/individual/{filename}', compress=True)


# sprint1 = {"sprint1": data['0-ACCAD_s009_Sprint1_poses']}

# print(sprint1.keys())

# print(sprint1['sprint1']['pose_quat_global'].shape)

# # first_seq = next(iter(data.values()))
# # print(first_seq['pose_quat_global'].shape)
# joblib.dump(sprint1, "data/amass/amass_sprint1.pkl", compress=True)


# # data = joblib.load('sample_data/amass_isaac_standing_upright_slim.pkl')

# # print(data.keys())

# # print(data['standing'].keys())