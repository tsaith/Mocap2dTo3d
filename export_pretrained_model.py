from __future__ import print_function, absolute_import, division

import torch
import torchvision

from src.model import LinearModel, weight_init


print("Start ---")

device = "cuda" # "cuda" or "cpu"

pretrained_model_path = "/home/andrew/projects/3d_pose_baseline_pytorch/models/ckpt_best.pth.tar"
print("pretrained_model_path : ", pretrained_model_path)

# An instance of your model
model = LinearModel()
model = model.cuda()
#model = model.cpu()

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
traced_script_module.save("pretained_model_cpp.pt")

print("End ---")