from __future__ import print_function, absolute_import, division

import os

import torch
import torchvision

from src.model import LinearModel, weight_init


print("Start ---")

device = "cuda" # "cuda" or "cpu"

work_dir_path = "/home/andrew/projects/MocapTo3d"
data_dir_name = "data"
test_2d_dataset_name = "test_2d.pth.tar"
test_3d_dataset_name = "test_3d.pth.tar"

test_2d_dataset_path = os.path.join(work_dir_path, data_dir_name, test_2d_dataset_name) 
test_3d_dataset_path = os.path.join(work_dir_path, data_dir_name, test_3d_dataset_name) 

print("test_2d_dataset_path: ", test_2d_dataset_path)
print("test_3d_dataset_path: ", test_3d_dataset_path)



print("End ---")