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
#from baseline_utils import get_dim_use_2d

import matplotlib.pyplot as plt
from plot_utils import plot_bl_inputs, plot_bl_outputs
from plot_utils import plot_bl_pose_2d, plot_bl_pose_3d

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
    bl_output = BaselineData()
    bl_target = BaselineData()


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
    use_dim_inputs = bl_data.get_use_dim_inputs()
    data_means = bl_data.get_data_means()
    data_stddevs = bl_data.get_data_stddevs()

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

    for i, (inps, tars) in enumerate(test_loader):

        if i == 0:

            print(f"shape of inps: {inps.shape}")


            inputs = Variable(inps.cuda())
            targets = Variable(tars.cuda())
            outputs = model(inputs)

            inputs = inputs.data.cpu().numpy()
            targets = targets.data.cpu().numpy()
            outputs = outputs.data.cpu().numpy()

            print("shape of inputs: ", inputs.shape)
            print("shape of targets: ", targets.shape)
            print("shape of outputs: ", outputs.shape)

            # calculate erruracy
            targets_unnorm = data_process.unNormalizeData(targets, stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])
            outputs_unnorm = data_process.unNormalizeData(outputs, stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])

            # remove dim ignored
            dim_use = np.hstack((np.arange(3), stat_3d['dim_use']))

            outputs_use = outputs_unnorm[:, dim_use]
            targets_use = targets_unnorm[:, dim_use]

            print(f"len(stat_3d['dim_use']): {len(stat_3d['dim_use'])}")
            print(f"stat_3d['dim_use']: {stat_3d['dim_use']}")

            print(f"len(stat_3d['mean']),: {len(stat_3d['mean'])}")
            print(f"len(stat_3d['std']): {len(stat_3d['std'])}")

            print(f"stat_3d.keys(): {stat_3d.keys()}")

            print(f"len(stat_3d['dim_ignore']): {len(stat_3d['dim_ignore'])}")
            print(f"stat_3d['dim_ignore']: {stat_3d['dim_ignore']}")


    data_input = inputs[0]
    data_output = outputs[0]

    print(f"data_input: {data_input}")

    bl_target.update_with_bl_pose(targets_use[0])
    bl_output.update_with_bl_pose(outputs_use[0])

    target_pose = bl_target.get_pose()
    output_pose = bl_output.get_pose()


    input_fig = plot_bl_inputs(data_input, title="2d baseline inputs")
    input_fig.savefig("data_input.jpg")

    output_fig = plot_bl_outputs(data_output, title="2d baseline outputs")
    output_fig.savefig("data_output.jpg")

    pose_2d_fig = plot_bl_pose_2d(bl_output, title="baseline pose 2d")
    pose_2d_fig.savefig("pose_2d.jpg")

    pose_3d_fig = plot_bl_pose_3d(bl_output, title="baseline pose 3d")
    pose_3d_fig.savefig("pose_3d.jpg")

    target_2d_fig = plot_bl_pose_2d(bl_target, title="baseline target 2d")
    target_2d_fig.savefig("target_2d.jpg")

    target_3d_fig = plot_bl_pose_3d(bl_target, title="baseline target 3d")
    target_3d_fig.savefig("target_3d.jpg")



if __name__ == "__main__":

    option = Options().parse()
    main(option)
