# Few-Shot Viewpoint Estimation

(ECCV 2020) PyTorch implementation of paper "Few-Shot Object Detection and Viewpoint Estimation for Objects in the Wild"\
[\[PDF\]](https://arxiv.org/abs/2007.12107) [\[Project webpage\]](http://imagine.enpc.fr/~xiaoy/FSDetView/) [\[Code (Detection)\]](https://github.com/YoungXIAO13/FewShotDetection)

<p align="center">
<img src="https://github.com/YoungXIAO13/FewShotViewpoint/blob/master/img/PipelineView.png" width="800px" alt="teaser">
</p>

If our project is helpful for your research, please consider citing:
```
@INPROCEEDINGS{Xiao2020FSDetView,
    author    = {Yang Xiao and Renaud Marlet},
    title     = {Few-Shot Object Detection and Viewpoint Estimation for Objects in the Wild},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year      = {2020}}
```

## Tabel of Contents
* [Installation](#installation)
* [Data Preparation](#data-preparation)
* [Getting Started](#getting-started)


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

Download [ObjectNet3D](https://cvgl.stanford.edu/projects/objectnet3d/):
```sh
cd ./data/ObjectNet3D
bash download_object3d.sh
```

Data structure should look like:
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

Download [Pascal3D](https://cvgl.stanford.edu/projects/pascal3d.html):
```sh
cd ./data/Pascal3D
bash download_pascal3d.sh
```

Data structure should look like:
```
data/Pascal3D
    Annotations/
    Images/
    ImageSets/
    Pointclouds/
    ...
```

## Getting Started

### Base-Class Training

We provide pre-trained models of **base-class training**:
```bash
bash download_models.sh
```
You will get a dir like:
```
save_models/
    IntraDataset/checkpoint.pth
    InterDataset/checkpoint.pth
```

You can also train the network yourself by running:
```bash
# Intra-Dataset
bash run/train_intra.sh

# Inter-Dataset
bash run/train_inter.sh
```

### Few-Shot Fine-tuning

Fine-tune the base-training models on a balanced training data including both base and novel classes:
```bash
bash run/finetune_intra.sh

bash run/finetune_inter.sh
```


### Testing

In **intra-dataset** setting, we test on the 20 novel classes of ObjectNet3D:
```bash
bash run/test_intra.sh
```

In **inter-dataset** setting, we train on the 12 novel classes of Pascal3D:
```bash
bash run/test_inter.sh
```

### Multiple Runs

Once the base-class training is done, you can run 10 times few-shot fine-tuning and testing with few-shot training data randomly selected for each run:
```bash
bash run/multiple_times_intra.sh

bash run/multiple_times_inter.sh
``` 

To get the performance averaged over multiple runs:
```bash
python mean_metrics.py save_models/IntraDataset_shot10

python mean_metrics.py save_models/InterDataset_shot10
``` 