# Few-Shot Viewpoint Estimation

(ECCV 2020) PyTorch implementation of paper "Few-Shot Object Detection and Viewpoint Estimation for Objects in the Wild"\
[\[PDF\]]() [\[Project webpage\]](http://imagine.enpc.fr/~xiaoy/FSDetView/)

<p align="center">
<img src="https://github.com/YoungXIAO13/FewShotDetection/blob/master/img/PipelineView.png" width="800px" alt="teaser">
</p>

If our project is helpful for your research, please consider citing:
```Bash
@INPROCEEDINGS{Xiao2020FSDetView,
    author    = {Yang Xiao and Renaud Marlet},
    title     = {Few-Shot Object Detetcion and Viewpoint Estimation for Objects in the Wild},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year      = {2020}}
```

## Tabel of Contents
* [Installation](#installation)
* [Data Preparation](#data-preparation)
* [Getting Start](#getting-start)


## Installation

Code built on top of [PoseFromShape](https://github.com/YoungXIAO13/PoseFromShape).
 
**Requirements**

* CUDA 8.0
* Python=3.6
* PyTorch>=0.4.1
* torchvision matched your PyTorch version

**Build**

Create conda env:
```sh
## Create conda env
conda create --name FSviewpoint --file spec-file.txt
conda activate FSviewpoint
conda install -c conda-forge matplotlib

## Install blender as a python module
conda install auxiliary/python-blender-2.77-py36_0.tar.bz2
```


## Data Preparation

We evaluate our method on two commonly-used benchmarks:

### ObjectNet3D (Intra-dataset)

We use the train set of ObjectNet3D for training and the val set for evaluation. 
Following [StarMap](https://github.com/xingyizhou/StarMap), we split the 100 object classes into 80 base classes and 20 novel classes. 

* Download [ObjectNet3D](https://cvgl.stanford.edu/projects/objectnet3d/):
```sh
cd ./data/ObjectNet3D
bash download_object3d.sh
```

* Data structure should look like:
```
data/ObjectNet3D
    Annotations/
    Images/
    ImageSets/
    Pointclouds/
    ...
```

### Pascal3D (Inter-dataset)

We use the train set of ObjectNet3D for training and the val set of Pascal3D for evaluation.
Following [MetaView](https://arxiv.org/abs/1905.04957), we use the 12 object classes that are the same with Pascal3D as novel classes and use the rest 88 as base classes.  

* Download [Pascal3D](https://cvgl.stanford.edu/projects/pascal3d.html):
```sh
cd ./data/Pascal3D
bash download_pascal3d.sh
```

* Data structure should look like:
```
data/Pascal3D
    Annotations/
    Images/
    ImageSets/
    Pointclouds/
    ...
```

## Getting Start


### Training


### Testing


**Testing results** will be writen in ``./save_models/{exp}/{dataset_name}/Kshots_out.txt``.
