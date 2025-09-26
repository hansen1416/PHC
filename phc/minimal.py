import glob
import os
import sys
import pdb
import os.path as osp
import joblib



sys.path.append(os.getcwd())


data = joblib.load("./root_states1.pkl")

root_states = data["root_states"]
dof_state = data["dof_state"]
dof_pos = data["dof_pos"]

print(root_states.shape)
print(dof_state.shape)
print(dof_pos.shape)

