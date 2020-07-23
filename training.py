import argparse
import numpy as np
import random
import os, sys
import time
import matplotlib
matplotlib.use('agg')  # use matplotlib without GUI support
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

sys.path.append('./auxiliary/')
from auxiliary.model import PoseEstimator
from auxiliary.dataset import Pascal3D
from auxiliary.utils import AverageValueMeter, load_checkpoint, KaiMingInit, rotation_acc, get_pred_from_cls_output
from auxiliary.loss import DeltaLoss, CELoss, SmoothCELoss
from evaluation import val

# =================PARAMETERS=============================== #
parser = argparse.ArgumentParser()

## training param
parser.add_argument('--resume', type=str, default=None, help='optional resume model path')
parser.add_argument('--save_dir', type=str, default='save_models', help='save directory')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate of optimizer')
parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--n_epoch', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr_step', type=int, default=200, help='step to decrease lr')
parser.add_argument('--print_freq', type=int, default=100, help='frequency of output print')

## dataset param
parser.add_argument('--setting', type=str, default=None, choices=['IntraDataset', 'InterDataset'])
parser.add_argument('--root_dir_train', type=str, default=None, help='training dataset directory')
parser.add_argument('--annot_train', type=str, default=None, help='training dataset annotation file')
parser.add_argument('--novel', action='store_true', help='whether to exclude novel classes during training')
parser.add_argument('--keypoint', action='store_true', help='use only samples with keypoint annotations')
parser.add_argument('--shot', type=int, default=None, help='few shot number')

## method param
parser.add_argument('--input_dim', type=int, default=224, help='input image dimension')
parser.add_argument('--img_feature_dim', type=int, default=512, help='feature dimension for images')
parser.add_argument('--point_num', type=int, default=2500, help='number of points used in each sample')
parser.add_argument('--shape_feature_dim', type=int, default=512, help='feature dimension for shapes')
parser.add_argument('--bin_size', type=int, default=15, help='bin size for the euler angle classification')
parser.add_argument('--smooth', type=float, default=0.2, help='activate label smoothing in classification')

opt = parser.parse_args()
print(opt)
# ========================================================== #


# =================CREATE DATASET=========================== #
if opt.novel:
    print('Excluding novel classes in training')
else:
    print('Including novel classes in training')

# 20 novel classes on ObjectNet3D
if opt.setting == 'IntraDataset':
    test_cats = ['bed', 'bookshelf', 'calculator', 'cellphone', 'computer', 'door', 'filing_cabinet', 'guitar', 'iron',
                 'knife', 'microwave', 'pen', 'pot', 'rifle', 'shoe', 'slipper', 'stove', 'toilet', 'tub', 'wheelchair']
    dataset_train = Pascal3D(train=True, root_dir=opt.root_dir_train, annotation_file=opt.annot_train,
                             cls_choice=test_cats, input_dim=opt.input_dim, point_num=opt.point_num,
                             keypoint=opt.keypoint, novel=opt.novel, shot=opt.shot)

# 12 novel classes on Pascal3D
elif opt.setting == 'InterDataset':
    test_cats = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car',
                 'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor']
    dataset_train = Pascal3D(train=True, root_dir=opt.root_dir_train, annotation_file=opt.annot_train,
                             cls_choice=test_cats, input_dim=opt.input_dim, point_num=opt.point_num,
                             keypoint=opt.keypoint, novel=opt.novel, shot=opt.shot)
else:
    sys.exit('Wrong setting!')

train_loader = DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, num_workers=opt.workers, drop_last=True)

print('train data consist of {} samples'.format(len(dataset_train)))
# ========================================================== #


# ================CREATE NETWORK============================ #
azi_classes, ele_classes, inp_classes = int(360 / opt.bin_size), int(180 / opt.bin_size), int(360 / opt.bin_size)

model = PoseEstimator(shape_feature_dim=opt.shape_feature_dim, img_feature_dim=opt.img_feature_dim,
                      azi_classes=azi_classes, ele_classes=ele_classes, inp_classes=inp_classes)
model.cuda()
if opt.resume is not None:
    load_checkpoint(model, opt.resume)
else:
    model.apply(KaiMingInit)
# ========================================================== #


# ================CREATE OPTIMIZER AND LOSS================= #
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, weight_decay=5e-4)
lrScheduler = optim.lr_scheduler.MultiStepLR(optimizer, [opt.lr_step], gamma=0.1)

criterion_azi = SmoothCELoss(360, 24, opt.smooth) if opt.smooth is not None else CELoss(360)
criterion_ele = SmoothCELoss(180, 12, opt.smooth) if opt.smooth is not None else CELoss(180)
criterion_inp = SmoothCELoss(360, 24, opt.smooth) if opt.smooth is not None else CELoss(360)
criterion_reg = DeltaLoss(opt.bin_size)
# ========================================================== #


# =============DEFINE stuff for logs ======================= #
result_path = os.path.join(os.getcwd(), opt.save_dir)
if not os.path.exists(result_path):
    os.makedirs(result_path)
logname = os.path.join(result_path, 'training_log.txt')
with open(logname, 'a') as f:
    f.write(str(opt) + '\n' + '\n')
    f.write('training set: ' + str(len(dataset_train)) + '\n')
    # f.write('evaluation set: ' + str(len(dataset_eval)) + '\n')

# arrays for saving the losses and accuracies
losses = np.zeros((opt.n_epoch, 2))  # training loss and validation loss
accuracies = np.zeros((opt.n_epoch, 2))  # train and val accuracy for classification and viewpoint estimation
# ========================================================== #


# =================== DEFINE TRAIN ========================= #
def train(data_loader, model, bin_size, criterion_azi, criterion_ele, criterion_inp, criterion_reg, optimizer):
    train_loss = AverageValueMeter()
    train_acc_rot = AverageValueMeter()

    model.train()

    data_time = AverageValueMeter()
    batch_time = AverageValueMeter()
    end = time.time()
    for i, data in enumerate(data_loader):
        # load data and label
        im, shapes, label, _ = data
        im, shapes, label = im.cuda(), shapes.cuda(), label.cuda()
        data_time.update(time.time() - end)

        # forward pass
        out = model(im, shapes)

        # compute losses and update the meters
        loss_azi = criterion_azi(out[0], label[:, 0])
        loss_ele = criterion_ele(out[1], label[:, 1])
        loss_inp = criterion_inp(out[2], label[:, 2])
        loss_reg = criterion_reg(out[3], out[4], out[5], label.float())
        loss = loss_azi + loss_ele + loss_inp + loss_reg
        train_loss.update(loss.item(), im.size(0))

        # compute rotation matrix accuracy
        preds = get_pred_from_cls_output([out[0], out[1], out[2]])
        for n in range(len(preds)):
            pred_delta = out[n + 3]
            delta_value = pred_delta[torch.arange(pred_delta.size(0)), preds[n].long()].tanh() / 2
            preds[n] = (preds[n].float() + delta_value + 0.5) * bin_size
        acc_rot = rotation_acc(torch.cat((preds[0].unsqueeze(1), preds[1].unsqueeze(1), preds[2].unsqueeze(1)), 1),
                               label.float())
        train_acc_rot.update(acc_rot.item(), im.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure bacth time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % opt.print_freq == 0:
            print("\tEpoch %3d --- Iter [%d/%d] Train loss: %.2f || Train accuracy: %.2f" %
                  (epoch, i + 1, len(data_loader), train_loss.avg, train_acc_rot.avg))
            print("\tData loading time: %.2f (%.2f)-- Batch time: %.2f (%.2f)\n" %
                  (data_time.val, data_time.avg, batch_time.val, batch_time.avg))

    return [train_loss.avg, train_acc_rot.avg]
# ========================================================== #


# =============BEGIN OF THE LEARNING LOOP=================== #
# initialization
best_acc = 0.

for epoch in range(opt.n_epoch):
    # update learning rate
    lrScheduler.step()

    # train
    train_loss, train_acc_rot = train(train_loader, model, opt.bin_size,
                                      criterion_azi, criterion_ele, criterion_inp, criterion_reg, optimizer)

    # save checkpoint
    torch.save({'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'losses': losses},
               os.path.join(result_path, 'checkpoint.pth'))

    # save losses and accuracies into log file
    with open(logname, 'a') as f:
        text = str('Epoch: %03d || train_loss %.2f || train_acc %.2f \n \n' % (epoch, train_loss, train_acc_rot))
        f.write(text)
# ========================================================== #


import pickle
import collections

data_loader = DataLoader(dataset_train, batch_size=1, shuffle=False, num_workers=1)

with torch.no_grad():
    model.eval()

    class_data = collections.defaultdict(list)
    for i, data in enumerate(data_loader):
        # load data and label
        im, shapes, label, cls = data
        im, shapes = im.cuda(), shapes.cuda()

        cls_data = model(im, shapes, get_mean_cls_data=True)

        class_data[cls[0]].append(cls_data.squeeze())

# calculate mean attention vectors of every class
mean_class_data = {k: sum(v) / len(v) for k, v in class_data.items()}

with open(os.path.join(result_path, 'mean_class_data.pkl'), 'wb') as f:
    pickle.dump(mean_class_data, f, pickle.HIGHEST_PROTOCOL)
print('save mean class data done!')
