import os
from os.path import join, basename
import numpy as np
from math import radians, cos, sin
from PIL import Image, ImageFilter
import pandas as pd
import cv2
import pymesh

import torch
import torchvision.transforms as transforms
import torch.utils.data as data








# ================================================= #
# Datasets used for training
# ================================================= #
class Pascal3D(data.Dataset):
    def __init__(self,
                 root_dir, annotation_file, input_dim=224, shape=None, shape_dir=None, view=None,
                 rotated=False, novel=True, keypoint=False, train=True, cat_choice=None,
                 point_num=2500, view_num=12, tour=2, random_range=0, random_model=False, shot=None):

        self.root_dir = root_dir
        self.input_dim = input_dim
        self.shape = shape
        self.shape_dir = shape_dir
        self.view = view
        self.point_num = point_num
        self.view_num = view_num
        self.train = train
        self.tour = tour
        self.rotated = rotated
        self.random_range = random_range
        self.random_model = random_model
        self.rotational_symmetry_cats = ['ashtray', 'basket', 'bottle', 'bucket', 'can', 'cap', 'cup',
                                         'fire_extinguisher', 'fish_tank', 'flashlight', 'helmet', 'jar', 'paintbrush',
                                         'pen', 'pencil', 'plate', 'pot', 'road_pole', 'screwdriver', 'toothbrush',
                                         'trash_bin', 'trophy']

        # load the data frame for annotations
        frame = pd.read_csv(os.path.join(root_dir, annotation_file))
        frame = frame[frame.elevation != 90]
        frame = frame[frame.difficult == 0]
        if annotation_file == 'ObjectNet3D.txt':
            if keypoint:
                frame = frame[frame.has_keypoints == 1]
                frame = frame[frame.truncated == 0]
                frame = frame[frame.occluded == 0]
            frame.azimuth = (360. + frame.azimuth) % 360
        if train:
            frame = frame[frame.set == 'train']
        else:
            frame = frame[frame.set == 'val']
            frame = frame[frame.truncated == 0]
            frame = frame[frame.occluded == 0]

        # choose cats for Object3D
        if cat_choice is not None:
            if train:
                frame = frame[~frame.cat.isin(cat_choice)] if novel else frame
            else:
                frame = frame[frame.cat.isin(cat_choice)]

        # sample K-shot training data
        if train and shot is not None:
            cats = np.unique(frame.cat)
            fewshot_frame = []
            for cat in cats:
                fewshot_frame.append(frame[frame.cat == cat].sample(n=shot))
            frame = pd.concat(fewshot_frame)

        self.annotation_frame = frame

        # define data augmentation and preprocessing for RGB images in training
        self.im_augmentation = transforms.Compose([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
            transforms.ToTensor(), normalize, disturb])

        # define data preprocessing for RGB images in validation
        self.im_transform = transforms.Compose([transforms.ToTensor(), normalize])

        # define data preprocessing for rendered multi view images
        self.render_transform = transforms.ToTensor()
        if input_dim != 224:
            self.render_transform = transforms.Compose([transforms.Resize(input_dim), transforms.ToTensor()])

    def __len__(self):
        return len(self.annotation_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotation_frame.iloc[idx, -1])
        cat = self.annotation_frame.iloc[idx]['cat']
        cad_index = self.annotation_frame.iloc[idx]['cad_index']

        # select a random shape from the same category in testing
        if self.random_model:
            df_cat = self.annotation_frame[self.annotation_frame.cat == cat]
            cad_index = np.random.choice(np.unique(df_cat.cad_index))

        left = self.annotation_frame.iloc[idx]['left']
        upper = self.annotation_frame.iloc[idx]['upper']
        right = self.annotation_frame.iloc[idx]['right']
        lower = self.annotation_frame.iloc[idx]['lower']

        # use continue viewpoint annotation
        label = self.annotation_frame.iloc[idx, 9:12].values

        # load real images in a Tensor of size C*H*W
        im = Image.open(img_name).convert('RGB')

        if self.train:
            # Gaussian blur
            if min(right - left, lower - upper) > 224 and np.random.random() < 0.3:
                im = im.filter(ImageFilter.GaussianBlur(3))

            # crop the original image with 2D bounding box jittering
            im = random_crop(im, left, upper, right - left, lower - upper)

            # Horizontal flip
            if np.random.random() > 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
                label[0] = 360 - label[0]
                label[2] = -label[2]

            # Random rotation
            if np.random.random() > 0.5:
                r = max(-60, min(60, np.random.randn() * 30))
                im = im.rotate(r)
                label[2] = label[2] + r
                label[2] += 360 if label[2] < -180 else (-360 if label[2] > 180 else 0)

            # pad it to the desired size
            im = resize_pad(im, self.input_dim)
            im = self.im_augmentation(im)
        else:
            # crop the ground truth bounding box and pad it to the desired size
            im = im.crop((left, upper, right, lower))
            im = resize_pad(im, self.input_dim)
            im = self.im_transform(im)

        label[0] = (360. - label[0]) % 360.
        label[1] = label[1] + 90.
        label[2] = (label[2] + 180.) % 360.
        label = label.astype('int')

        if self.shape is None:
            label = torch.from_numpy(label).long()
            return im, label

        # randomize the canonical view with respect to the azimuth
        # range_0: [-45, 45]; range_1: [-90, 90]; range_2: [-180, 180]
        if self.rotated and cat not in self.rotational_symmetry_cats:
            rotation = np.random.randint(-8, 9) % 72 if self.random_range == 0 else \
                (np.random.randint(-17, 18) % 72 if self.random_range == 1 else np.random.randint(0, 72))
            label[0] = (label[0] - rotation * 5) % 360
        else:
            rotation = 0

        if self.shape == 'nontextured' or self.shape == 'nocs':

            # load render images in a Tensor of size K*C*H*W
            render_path = os.path.join(self.root_dir, self.shape_dir, self.view, cat, '{:02d}'.format(cad_index), self.shape)

            # read multiview rendered images
            if self.view == 'semisphere':
                renders = read_semisphere(self.render_transform, render_path, self.view_num, self.tour, rotation)
            else:
                renders = read_dodecahedron(self.render_transform, render_path, self.view_num, rotation)

            label = torch.from_numpy(label).long()
            if self.train:
                return im, renders, label, cat
            else:
                return im, renders, label

        if self.shape == 'pointcloud':

            # load point_clouds
            pointcloud_path = os.path.join(self.root_dir, self.shape_dir, cat, '{:02d}'.format(cad_index), 'compressed.ply')
            point_cloud = read_pointcloud(pointcloud_path, self.point_num, rotation)

            if self.train:
                return im, point_cloud, label, cat
            else:
                return im, point_cloud, label



if __name__ == "__main__":
    
    test_cats = ['bed', 'bookshelf', 'calculator', 'cellphone', 'computer', 'door', 'filing_cabinet', 'guitar', 'iron', 'knife', 'microwave', 'pen', 'pot', 'rifle', 'shoe', 'slipper', 'stove', 'toilet', 'tub', 'wheelchair']

    d = Pascal3D(root_dir='/home/xiao/Datasets/Pascal3D', annotation_file='Pascal3D.txt', 
                 shape_dir='pointcloud', shape='pointcloud', view='dodecahedron', view_num=12, point_num=2500,
                 rotated=False, train=True, keypoint=True, cat_choice=None)

    print('length is %d' % len(d))

    from torch.utils.data import DataLoader
    import sys
    import time
    test_loader = DataLoader(d, batch_size=1, shuffle=True)
    begin = time.time()
    for i, data in enumerate(test_loader):
        im, shape, label, cls = data
        print(time.time() - begin)
        if i == 0:
            print(im.size(), shape.size(), label.size())
            print(cls[0])
            sys.exit()
