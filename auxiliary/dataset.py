import os
import numpy as np
from PIL import Image, ImageFilter
import pandas as pd
import pymesh

import torch
import torchvision.transforms as transforms
import torch.utils.data as data


# Lighting noise transform
class TransLightning(object):
    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone()\
            .mul(alpha.view(1, 3).expand(3, 3))\
            .mul(self.eigval.view(1, 3).expand(3, 3))\
            .sum(1).squeeze()
        return img.add(rgb.view(3, 1, 1).expand_as(img))


# ImageNet statistics
imagenet_pca = {
        'eigval': torch.Tensor([0.2175, 0.0188, 0.0045]),
        'eigvec': torch.Tensor([[-0.5675, 0.7192, 0.4009],
                                [-0.5808, -0.0045, -0.8140],
                                [-0.5836, -0.6948, 0.4203],
                                ])
}


# Define normalization and random disturb for input image
disturb = TransLightning(0.1, imagenet_pca['eigval'], imagenet_pca['eigvec'])
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def random_crop(im, x, y, w, h):
    left = max(0, x + int(np.random.uniform(-0.1, 0.1) * w))
    upper = max(0, y + int(np.random.uniform(-0.1, 0.1) * h))
    right = min(im.size[0], x + int(np.random.uniform(0.9, 1.1) * w))
    lower = min(im.size[1], y + int(np.random.uniform(0.9, 1.1) * h))
    im_crop = im.crop((left, upper, right, lower))
    return im_crop


def resize_pad(im, dim):
    w, h = im.size
    im = transforms.functional.resize(im, int(dim * min(w, h) / max(w, h)))
    left = int(np.ceil((dim - im.size[0]) / 2))
    right = int(np.floor((dim - im.size[0]) / 2))
    top = int(np.ceil((dim - im.size[1]) / 2))
    bottom = int(np.floor((dim - im.size[1]) / 2))
    im = transforms.functional.pad(im, (left, top, right, bottom))
    return im


def read_pointcloud(model_path, point_num):
    # read in original point cloud
    point_cloud_raw = pymesh.load_mesh(model_path).vertices

    # randomly select a fix number of points on the surface
    point_subset = np.random.choice(point_cloud_raw.shape[0], point_num, replace=False)
    point_cloud = point_cloud_raw[point_subset]
    point_cloud = torch.from_numpy(point_cloud.transpose()).float()

    # normalize the point cloud into [0, 1]
    point_cloud = point_cloud - torch.min(point_cloud)
    point_cloud = point_cloud / torch.max(point_cloud)

    return point_cloud


# ================================================= #
# Datasets used for training
# ================================================= #
class Pascal3D(data.Dataset):
    def __init__(self, root_dir, annotation_file, input_dim=224, point_num=2500, train=True,
                 keypoint=True, novel=True, cls_choice=None, shot=None):
        self.train = train
        self.root_dir = root_dir
        self.input_dim = input_dim
        self.point_num = point_num

        # load the data frame for annotations
        frame = pd.read_csv(os.path.join(root_dir, annotation_file))
        frame = frame[frame.elevation != 90]
        frame = frame[frame.difficult == 0]
        if annotation_file == 'ObjectNet3D.txt':
            frame.azimuth = (360. + frame.azimuth) % 360

        # evaluation only on non-occluded and non-truncated objects with keypoint annotations as MetaView/StarMap
        if train:
            frame = frame[frame.set == 'train']
        else:
            frame = frame[frame.set == 'val']
            frame = frame[frame.truncated == 0]
            frame = frame[frame.occluded == 0]
            frame = frame[frame.has_keypoints == 1]

        # we exclude training samples without keypoint annotations for a fair comparison with MetaView
        if train and keypoint:
            frame = frame[frame.has_keypoints == 1]

        # exclude novel classes for training and include it for testing
        if cls_choice is not None:
            if train:
                frame = frame[~frame.cat.isin(cls_choice)] if novel else frame
            else:
                frame = frame[frame.cat.isin(cls_choice)]

        # sample K-shot training data
        if train and shot is not None:
            frame = frame[frame.truncated == 0]
            frame = frame[frame.occluded == 0]
            categories = np.unique(frame.cat)
            fewshot_frame = []
            for cat in categories:
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
        # get image path and category
        img_path = os.path.join(self.root_dir, self.annotation_frame.iloc[idx, -1])
        cls = self.annotation_frame.iloc[idx]['cat']
        
        # randomly choose a shape or an exemplar shape
        cls_index = np.unique(self.annotation_frame[self.annotation_frame.cat == cls].cad_index)
        cad_index = np.random.choice(cls_index)

        left = self.annotation_frame.iloc[idx]['left']
        upper = self.annotation_frame.iloc[idx]['upper']
        right = self.annotation_frame.iloc[idx]['right']
        lower = self.annotation_frame.iloc[idx]['lower']

        # use continue viewpoint annotation
        label = self.annotation_frame.iloc[idx, 9:12].values

        # load real images in a Tensor of size C*H*W
        im = Image.open(img_path).convert('RGB')

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

        # transform the ground-truth angle values into [0, 360)
        label[0] = (360. - label[0]) % 360.
        label[1] = label[1] + 90.
        label[2] = (label[2] + 180.) % 360.
        label = label.astype('int')

        # load point_clouds
        pointcloud_path = os.path.join(self.root_dir, 'Pointclouds', cls, '{:02d}'.format(cad_index), 'compressed.ply')
        point_cloud = read_pointcloud(pointcloud_path, self.point_num)

        if self.train:
            return im, point_cloud, label, cls
        else:
            return im, point_cloud, label
