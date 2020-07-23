import numpy as np
import os, sys
from os.path import join, dirname
import argparse
from tqdm import tqdm
import pandas as pd
import pickle

import matplotlib
matplotlib.use('agg')  # use matplotlib without GUI support

sys.path.append('./auxiliary/')
from auxiliary.model import PoseEstimator
from auxiliary.dataset import Pascal3D
from auxiliary.utils import load_checkpoint
from evaluation import test_category

# =================PARAMETERS=============================== #
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None, help='model path')
parser.add_argument('--class_data', type=str, default=None, help='offline computed mean class data path')
parser.add_argument('--output', type=str, default='results', help='testing results save path')

parser.add_argument('--setting', type=str, default=None, choices=['IntraDataset', 'InterDataset'])
parser.add_argument('--root_dir', type=str, default=None, help='dataset directory')
parser.add_argument('--input_dim', type=int, default=224, help='input image dimension')
parser.add_argument('--point_num', type=int, default=2500, help='number of points used in each sample')
parser.add_argument('--img_feature_dim', type=int, default=512, help='feature dimension for images')
parser.add_argument('--shape_feature_dim', type=int, default=512, help='feature dimension for shapes')
parser.add_argument('--bin_size', type=int, default=15, help='bin size for the euler angle classification')

opt = parser.parse_args()
print(opt)
# ========================================================== #


# ================CREATE NETWORK============================ #
azi_classes, ele_classes, inp_classes = int(360 / opt.bin_size), int(180 / opt.bin_size), int(360 / opt.bin_size)

model = PoseEstimator(shape_feature_dim=opt.shape_feature_dim, img_feature_dim=opt.img_feature_dim,
                      azi_classes=azi_classes, ele_classes=ele_classes, inp_classes=inp_classes)
model.cuda()
load_checkpoint(model, opt.model)
# ========================================================== #


# =============DEFINE stuff for logs ======================= #
# write basic information into the log file
if not os.path.isdir(opt.output):
    os.mkdir(opt.output)
logname = os.path.join(opt.output, 'testing.txt')

f = open(logname, mode='w')
f.write('\n')
f.close()
# ========================================================== #


if opt.setting == 'IntraDataset':
    annotation_file = 'ObjectNet3D.txt'
    test_cls = ['bed', 'bookshelf', 'calculator', 'cellphone', 'computer', 'door', 'filing_cabinet', 'guitar', 'iron',
                'knife', 'microwave', 'pen', 'pot', 'rifle', 'shoe', 'slipper', 'stove', 'toilet', 'tub', 'wheelchair']

elif opt.setting == 'InterDataset':
    annotation_file = 'Pascal3D.txt'
    test_cls = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car',
                'chair', 'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor']

else:
    sys.exit('Wrong setting!')

mean_class_data = pickle.load(open(opt.class_data, 'rb'))
Err_All = []

for cls in test_cls:
    class_data = mean_class_data[cls]
    dataset_test = Pascal3D(train=False, root_dir=opt.root_dir, annotation_file=annotation_file,
                            cls_choice=[cls], input_dim=opt.input_dim, point_num=opt.point_num)
    Acc, Med, Errs = test_category(dataset_test, model, opt.bin_size, cls, opt.output, logname, class_data)
    Err_All.extend(Errs)
    print('Acc30 is {:.2f} and MerErr is {:.2f} for {} images in class {}\n'.format(Acc, Med, len(dataset_test), cls))

print('Performance across all classes: Acc_pi/6 is {:.2f} and Med_Err is {:.2f}'.format(
    np.mean(np.array(Err_All) <= 30), np.median(np.array(Err_All))
))