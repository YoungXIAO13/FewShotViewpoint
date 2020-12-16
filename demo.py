import numpy as np
import os, sys
import argparse
import pickle
from PIL import Image
import matplotlib
matplotlib.use('agg')  # use matplotlib without GUI support

import torch
import torchvision.transforms as transforms

sys.path.append('./auxiliary/')
from auxiliary.model import PoseEstimator
from auxiliary.utils import load_checkpoint, get_pred_from_cls_output


# =================PARAMETERS=============================== #
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None, help='model path')
parser.add_argument('--class_data', type=str, default=None, help='offline computed mean class data path')

parser.add_argument('--test_cls', type=str, default=None)
parser.add_argument('--test_img', type=str, default=None)

parser.add_argument('--input_dim', type=int, default=224, help='input image dimension')
parser.add_argument('--img_feature_dim', type=int, default=512, help='feature dimension for images')
parser.add_argument('--shape_feature_dim', type=int, default=512, help='feature dimension for shapes')
parser.add_argument('--bin_size', type=int, default=15, help='bin size for the euler angle classification')

opt = parser.parse_args()
# ========================================================== #


# ================CREATE NETWORK============================ #
azi_classes, ele_classes, inp_classes = int(360 / opt.bin_size), int(180 / opt.bin_size), int(360 / opt.bin_size)

model = PoseEstimator(shape_feature_dim=opt.shape_feature_dim, img_feature_dim=opt.img_feature_dim,
                      azi_classes=azi_classes, ele_classes=ele_classes, inp_classes=inp_classes)
model.cuda()
load_checkpoint(model, opt.model)
model.eval()
# ========================================================== #


# ================LOAD CLASS FEATURES======================== #
mean_class_data = pickle.load(open(opt.class_data, 'rb'))
if opt.test_cls not in mean_class_data.keys():
    raise ValueError
cls_data = mean_class_data[opt.test_cls]
# =========================================================== #


# ======================GET INPUT IMAGE====================== #
def resize_pad(im, dim):
    w, h = im.size
    im = transforms.functional.resize(im, int(dim * min(w, h) / max(w, h)))
    left = int(np.ceil((dim - im.size[0]) / 2))
    right = int(np.floor((dim - im.size[0]) / 2))
    top = int(np.ceil((dim - im.size[1]) / 2))
    bottom = int(np.floor((dim - im.size[1]) / 2))
    im = transforms.functional.pad(im, (left, top, right, bottom))
    return im


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
im_transform = transforms.Compose([transforms.ToTensor(), normalize])

im = Image.open(opt.test_img).convert('RGB')
im = resize_pad(im, opt.input_dim)
im = im_transform(im)
# ========================================================== #


with torch.no_grad():
    im = im.cuda()
    im = im.unsqueeze(0)

    # forward pass
    out = model(im, None, mean_class_data=cls_data)

    # transform the output into the label format
    preds = get_pred_from_cls_output([out[0], out[1], out[2]])
    for n in range(len(preds)):
        pred_delta = out[n + 3]
        delta_value = pred_delta[torch.arange(pred_delta.size(0)), preds[n].long()].tanh() / 2
        preds[n] = (preds[n].float() + delta_value + 0.5) * opt.bin_size

    # Azimuth is between [0, 360), Elevation is between (-90, 90), In-plane Rotation is between [-180, 180)
    azi = preds[0].squeeze().cpu().numpy()
    ele = (preds[1] - 90).squeeze().cpu().numpy()
    rot = (preds[2] - 180).squeeze().cpu().numpy()
    print("Azimuth = {:.3f} \t Elevation = {:.3f} \t Inplane-Rotation = {:.3f}".format(azi, ele, rot))
