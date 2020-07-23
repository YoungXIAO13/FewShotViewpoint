gpu=1
save_dir="save_models/IntraDataset"

# few-shot fine-tuning
CUDA_VISIBLE_DEVICES=$gpu python -W ignore training.py --setting IntraDataset \
--root_dir_train data/ObjectNet3D --annot_train ObjectNet3D.txt \
--resume ${save_dir}/checkpoint.pth --save_dir ${save_dir}_shot10 \
--n_epoch 100 --lr_step 50 --keypoint --shot 10