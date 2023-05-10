#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import os
import sys
import time
from pprint import pprint
import numpy as np

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from opt import Options
from src.procrustes import get_transformation
import src.data_process as data_process
from src import Bar
import src.utils as utils
import src.misc as misc
import src.log as log

from src.model import LinearModel, weight_init
from src.datasets.human36m import Human36M

from baseline_data import BaselineData

from file_utils import make_dir
from plot_utils import plot_bl_inputs
from plot_utils import plot_bl_pose_2d, plot_bl_pose_3d
from anim_utils import make_animations


def add_hip(targets, is_3d=True):

    outputs = []
    dims = 3
    for target in targets:

        dims = 3 if is_3d else 2
        out = np.concatenate([np.zeros(dims), target], axis=0)
        outputs.append(out)

    return outputs

def main(opt):

    start_epoch = 0
    err_best = 1000
    glob_step = 0
    lr_now = opt.lr

    # save options
    log.save_options(opt, opt.ckpt)

    # create model
    print(">>> creating model")
    model = LinearModel()
    model = model.cuda()
    model.apply(weight_init)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    criterion = nn.MSELoss(size_average=True).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    bl_data = BaselineData()

    # load ckpt
    if opt.load:
        print(">>> loading ckpt from '{}'".format(opt.load))
        ckpt = torch.load(opt.load)
        start_epoch = ckpt['epoch']
        err_best = ckpt['err']
        glob_step = ckpt['step']
        lr_now = ckpt['lr']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(">>> ckpt loaded (epoch: {} | err: {})".format(start_epoch, err_best))
    if opt.resume:
        logger = log.Logger(os.path.join(opt.ckpt, 'log.txt'), resume=True)
    else:
        logger = log.Logger(os.path.join(opt.ckpt, 'log.txt'))
        logger.set_names(['epoch', 'lr', 'loss_train', 'loss_test', 'err_test'])

    # list of action(s)
    actions = misc.define_actions(opt.action)
    num_actions = len(actions)
    print(">>> actions to use (total: {}):".format(num_actions))
    pprint(actions, indent=4)

    # data loading
    # load statistics data
    stat_3d = torch.load(os.path.join(opt.data_dir, 'stat_3d.pth.tar'))

    bl_data.set_stat_3d(stat_3d)
    pose_means = bl_data.get_pose_means()
    pose_stddevs = bl_data.get_pose_stddevs()

    # test
    if opt.test:

        err_set = []
        action = actions[0]

        test_loader = DataLoader(
            dataset=Human36M(actions=action, data_path=opt.data_dir, use_hg=opt.use_hg, is_train=False),
            batch_size=opt.test_batch,
            shuffle=False,
            num_workers=opt.job,
            pin_memory=True)

    model.eval()

    action_index = 0
    for i, (inps, tars) in enumerate(test_loader):

        if i == action_index:

            inputs = Variable(inps.cuda())
            targets = Variable(tars.cuda())
            outputs = model(inputs)

            inputs = inputs.data.cpu().numpy()
            targets = targets.data.cpu().numpy()
            outputs = outputs.data.cpu().numpy()

            # calculate erruracy
            targets_unnorm = data_process.unNormalizeData(targets, stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])
            outputs_unnorm = data_process.unNormalizeData(outputs, stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])

            # remove dim ignored
            dim_use = np.hstack((np.arange(3), stat_3d['dim_use']))

            outputs_unnorm = outputs_unnorm[:, dim_use]
            targets_unnorm = targets_unnorm[:, dim_use]


    print(f"action_index: {action_index}") 
    print(f"frame_num: {len(targets)}") 

    # Add hip
    inputs = add_hip(inputs, is_3d=False)
    outputs = add_hip(outputs, is_3d=True)
    targets = add_hip(targets, is_3d=True)

    inputs_data = []
    for input in inputs:

        input_data = BaselineData()
        input_data.update_with_bl_pose_2d(input)
        inputs_data.append(input_data)


    outputs_data = []
    for output in outputs:

        output_data = BaselineData()
        output_data.update_with_bl_pose_3d(output)
        outputs_data.append(output_data)

    targets_data = []
    for target in targets:

        target_data = BaselineData()
        target_data.update_with_bl_pose_3d(target)
        targets_data.append(target_data)


    outputs_pose_data = []
    for output_data in outputs_data:

        pose = output_data.unnormalize(pose_means, pose_stddevs)
        output_pose_data = BaselineData()
        output_pose_data.update_with_pose(pose)

        outputs_pose_data.append(output_pose_data)

    targets_pose_data = []
    for target_data in targets_data:

        pose = target_data.unnormalize(pose_means, pose_stddevs)
        target_pose_data = BaselineData()
        target_pose_data.update_with_pose(pose)

        targets_pose_data.append(target_pose_data)

    
    targets_unnorm_data = []
    for target_unnorm in targets_unnorm:

        target_unnorm_data = BaselineData()
        target_unnorm_data.update_with_bl_pose_3d(target_unnorm)

        targets_unnorm_data.append(target_unnorm_data)

    inputs_unnorm_data = []
    for input_data in inputs_data:

        pose = input_data.unnormalize(pose_means, pose_stddevs, is_3d=False)
        input_unnorm_data = BaselineData()
        input_unnorm_data.update_with_pose(pose, is_3d=False)

        inputs_unnorm_data.append(input_unnorm_data)


    # Target unnorm
    fig = plot_bl_pose_2d(target_unnorm_data, title="baseline target 2d")
    fig.savefig("plot_target_unnorm_2d.jpg")

    fig = plot_bl_pose_3d(target_unnorm_data, title="baseline target 3d")
    fig.savefig("plot_target_unnorm_3d.jpg")

    # Make animations
    dir_path = "output/anims/inputs"
    make_animations(dir_path, inputs_data, is_3d=False, xlim=[-2, 2], ylim=[-2, 2])

    dir_path = "output/anims/inputs_unnorm"
    make_animations(dir_path, inputs_unnorm_data, is_3d=False)

    dir_path = "output/anims/outputs_pose"
    make_animations(dir_path, outputs_pose_data)

    dir_path = "output/anims/outputs_pose_2d"
    make_animations(dir_path, outputs_pose_data, is_3d=False)

    dir_path = "output/anims/targets_pose"
    make_animations(dir_path, targets_pose_data)

    dir_path = "output/anims/targets_pose_2d"
    make_animations(dir_path, targets_pose_data, is_3d=False)

    dir_path = "output/anims/target_unnorm"
    make_animations(dir_path, targets_unnorm_data)



if __name__ == "__main__":

    option = Options().parse()
    main(option)
