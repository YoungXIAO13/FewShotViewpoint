gpu=1
save_dir="save_models/InterDataset"

# base class training
CUDA_VISIBLE_DEVICES=$gpu python -W ignore training.py --setting InterDataset \
--root_dir_train /home/xiao/Datasets/ObjectNet3D --annot_train ObjectNet3D.txt \
--save_dir ${save_dir} \
--n_epoch 150 --novel

# few-shot fine-tuning
CUDA_VISIBLE_DEVICES=$gpu python -W ignore training.py --setting InterDataset \
--root_dir_train /home/xiao/Datasets/ObjectNet3D --annot_train ObjectNet3D.txt \
--resume ${save_dir}/checkpoint.pth --save_dir ${save_dir}_shot10 \
--n_epoch 100 --lr_step 50 --shot 10