gpu=1
save_dir="save_models/IntraDataset_shot10_1"

CUDA_VISIBLE_DEVICES=$gpu python testing.py \
--setting IntraDataset --root_dir /home/xiao/Datasets/ObjectNet3D/ \
--model ${save_dir}/checkpoint.pth \
--class_data ${save_dir}/mean_class_data.pkl \
--output ${save_dir}/pred_novel
