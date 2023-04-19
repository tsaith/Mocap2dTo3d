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

import matplotlib.pyplot as plt
from plot_utils import plot_baseline_data_2d, plot_baseline_pose_2d
from plot_utils import plot_baseline_pose_2d, plot_baseline_pose_3d

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

            print(f"len(stat_3d['mean']),: {len(stat_3d['mean'])}")
            print(f"len(stat_3d['std']): {len(stat_3d['std'])}")
            print(f"len(stat_3d['dim_use']): {len(stat_3d['dim_use'])}")

    data_input = inputs[0]
    data_output = outputs[0]

    target_3d = targets_use[0] 
    output_3d = outputs_use[0] 
 
    print(f"len(data_input): {len(data_input)}")
    print(f"len(data_output): {len(data_output)}")
    print(f"len(target_3d): {len(target_3d)}")
    print(f"len(output_3d): {len(output_3d)}")

    input_fig = plot_baseline_data_2d(data_input, title="2d baseline inputs")
    input_fig.savefig("data_input.jpg")

    output_fig = plot_baseline_data_2d(data_output, title="2d baseline outputs")
    output_fig.savefig("data_output.jpg")

    pose_2d_fig = plot_baseline_pose_2d(output_3d, title="baseline pose")
    pose_2d_fig.savefig("pose_2d.jpg")

    pose_3d_fig = plot_baseline_pose_3d(output_3d, title="baseline pose 3d")
    pose_3d_fig.savefig("pose_3d.jpg")

    target_3d_fig = plot_baseline_pose_3d(target_3d, title="baseline target 3d")
    target_3d_fig.savefig("target_3d.jpg")



def test(test_loader, model, criterion, stat_3d, procrustes=False):

    losses = utils.AverageMeter()

    model.eval()

    all_dist = []
    start = time.time()
    batch_time = 0
    bar = Bar('>>>', fill='>', max=len(test_loader))

    for i, (inps, tars) in enumerate(test_loader):

        print("")
        print("shape of inps: ", inps.shape)
        print("shape of tars: ", tars.shape)

        inputs = Variable(inps.cuda())
        #targets = Variable(tars.cuda(async=True))
        targets = Variable(tars.cuda())

        outputs = model(inputs)

        # calculate loss
        outputs_coord = outputs
        loss = criterion(outputs_coord, targets)

        losses.update(loss.item(), inputs.size(0))

        tars = targets

        # calculate erruracy
        targets_unnorm = data_process.unNormalizeData(tars.data.cpu().numpy(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])
        outputs_unnorm = data_process.unNormalizeData(outputs.data.cpu().numpy(), stat_3d['mean'], stat_3d['std'], stat_3d['dim_use'])

        # remove dim ignored
        dim_use = np.hstack((np.arange(3), stat_3d['dim_use']))

        outputs_use = outputs_unnorm[:, dim_use]
        targets_use = targets_unnorm[:, dim_use]

        if procrustes:
            for ba in range(inps.size(0)):
                gt = targets_use[ba].reshape(-1, 3)
                out = outputs_use[ba].reshape(-1, 3)
                _, Z, T, b, c = get_transformation(gt, out, True)
                out = (b * out.dot(T)) + c
                outputs_use[ba, :] = out.reshape(1, 51)

        sqerr = (outputs_use - targets_use) ** 2

        distance = np.zeros((sqerr.shape[0], 17))
        dist_idx = 0
        for k in np.arange(0, 17 * 3, 3):
            distance[:, dist_idx] = np.sqrt(np.sum(sqerr[:, k:k + 3], axis=1))
            dist_idx += 1
        all_dist.append(distance)

        # update summary
        if (i + 1) % 100 == 0:
            batch_time = time.time() - start
            start = time.time()

        bar.suffix = '({batch}/{size}) | batch: {batchtime:.4}ms | Total: {ttl} | ETA: {eta:} | loss: {loss:.6f}' \
            .format(batch=i + 1,
                    size=len(test_loader),
                    batchtime=batch_time * 10.0,
                    ttl=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg)
        bar.next()

    all_dist = np.vstack(all_dist)
    joint_err = np.mean(all_dist, axis=0)
    ttl_err = np.mean(all_dist)
    bar.finish()
    print (">>> error: {} <<<".format(ttl_err))
    return losses.avg, ttl_err


if __name__ == "__main__":

    option = Options().parse()
    main(option)
