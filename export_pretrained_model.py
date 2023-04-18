from __future__ import print_function, absolute_import, division

import os

import torch
import torchvision

from src.model import LinearModel, weight_init


print("Start ---")

device = "cuda" # "cuda" or "cpu"

work_dir_path = "/home/andrew/projects/MocapTo3d"
model_dir_name = "models"
pretrained_model_name = "ckpt_best.pth.tar"
exported_model_name = "pretained_model_cpp.pt"
pretrained_model_path = os.path.join(work_dir_path, model_dir_name, pretrained_model_name) 
print("pretrained_model_path : ", pretrained_model_path)

# An instance of your model
model = LinearModel()
model.cuda()
#model = model.cpu()
model.eval() 

# Load the pretrained model
ckpt = torch.load(pretrained_model_path)
model.load_state_dict(ckpt['state_dict'])

# Parameters

num_epochs = ckpt['epoch']
err_best = ckpt['err']
glob_step = ckpt['step']
lr = ckpt['lr']

batch_size = 2
input_size = model.input_size
output_size = model.output_size

print("num_epochs: ", num_epochs)
print("batch_size: ", batch_size)
print("input_size: ", input_size)
print("output_size: ", output_size)
print("lr_final: ", lr)

inputs = torch.rand([batch_size, input_size]).cuda()
print("shape of inputs: ", inputs.shape)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing
traced_script_module = torch.jit.trace(model, inputs)

# Output and Input
outputs = traced_script_module(inputs)
print("shape of outputs: ", outputs.shape)

print(type(outputs), outputs[0,:10], outputs.shape)

# This will produce a traced_resnet_model.pt file
# in working dir
print("Export the pretained model for c++ as ", exported_model_name)
traced_script_module.save(exported_model_name)

print("End ---")