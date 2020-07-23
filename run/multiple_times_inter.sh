gpu=1
save_dir="save_models/InterDataset"

# base class training
CUDA_VISIBLE_DEVICES=$gpu python -W ignore training.py --setting InterDataset \
--root_dir_train data/ObjectNet3D --annot_train ObjectNet3D.txt \
--save_dir ${save_dir} \
--n_epoch 150 --novel

# few-shot fine-tuning
for i in {1..10}
do
CUDA_VISIBLE_DEVICES=$gpu python -W ignore training.py --setting InterDataset \
--root_dir_train data/ObjectNet3D --annot_train ObjectNet3D.txt \
--resume ${save_dir}/checkpoint.pth --save_dir ${save_dir}_shot10_run${i} \
--n_epoch 100 --lr_step 50 --shot 10

CUDA_VISIBLE_DEVICES=$gpu python testing.py \
--setting InterDataset --root_dir data/Pascal3D/ \
--model ${save_dir}_shot10_run${i}/checkpoint.pth \
--class_data ${save_dir}_shot10_run${i}/mean_class_data.pkl \
--output ${save_dir}_shot10_run${i}/pred_novel
done