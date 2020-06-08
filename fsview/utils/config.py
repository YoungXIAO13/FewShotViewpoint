from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C


#
# Dataset split
#
__C.OBJECT3D_CLASSES = []

__C.NOVEL_CLASSES = []


__C.SYM_CLASSES = ['ashtray', 'basket', 'bottle', 'bucket', 'can', 'cap', 'cup',
                   'fire_extinguisher', 'fish_tank', 'flashlight', 'helmet', 'jar', 'paintbrush',
                   'pen', 'pencil', 'plate', 'pot', 'road_pole', 'screwdriver', 'toothbrush', 'trash_bin', 'trophy']

__C.PASCAL3D_CLASSES = ['aeroplane', 'bicycle', 'boat', 'bottle', 'bus', 'car', 'chair',
                        'diningtable', 'motorbike', 'sofa', 'train', 'tvmonitor']


#
# Encoding parameters
#
__C.IMAGE_DIM = 224

__C.SHAPE = 'pointcloud' # pointcloud, nocs, nontextured, depth
__C.SHAPE_DIR = 'pointcloud' # multiviews

__C.IMAGE_CHANNEL = 3
__C.SHAPE_CHANNEL = 3

__C.POINT_NUM = 2500

__C.RENDER_DIM = 224
__C.RENDER_NUM = 12
__C.RENDER_TOUR = 2



#
# Training options
#
__C.TRAIN = edict()

# Shot size
__C.TRAIN.SHOT = 0

# Randomize canonical azimuth and its random range
__C.TRAIN.ROTATE_AZI = True
__C.TRAIN.ROTATE_RANGE = [-45, 45]

# Only use training samples with keypoint annotations
__C.TRAIN.KEYPOINT = True

# Initial learning rate
__C.TRAIN.LEARNING_RATE = 0.001

# Momentum
__C.TRAIN.MOMENTUM = 0.9

# Weight decay, for regularization
__C.TRAIN.WEIGHT_DECAY = 0.0005

# Factor for reducing the learning rate
__C.TRAIN.GAMMA = 0.1

# Step size for reducing the learning rate, currently only support one step
__C.TRAIN.STEPSIZE = [30000]

# Iteration intervals for showing the loss during training, on command line interface
__C.TRAIN.DISPLAY = 10


#
# MISC
#

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))




def get_output_dir(imdb, weights_filename):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    outdir = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, imdb.name))
    if weights_filename is None:
        weights_filename = 'default'
    outdir = osp.join(outdir, weights_filename)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    return outdir


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        old_type = type(b[k])
        if old_type is not type(v):
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            else:
                raise ValueError(('Type mismatch ({} vs. {}) for config key: {}').format(type(b[k]), type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print(('Error under config key: {}'.format(k)))
                raise
        else:
            b[k] = v


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d
        try:
            value = literal_eval(v)
        except:
            # handle the case when v is a string literal
            value = v
        assert type(value) == type(d[subkey]), 'type {} does not match original type {}'.format(type(value), type(d[subkey]))
        d[subkey] = value