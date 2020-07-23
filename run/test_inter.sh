gpu=0
save_dir="save_models/InterDataset_shot10"

CUDA_VISIBLE_DEVICES=$gpu python testing.py \
--setting InterDataset --root_dir data/Pascal3D/ \
--model ${save_dir}/checkpoint.pth \
--class_data ${save_dir}/mean_class_data.pkl \
--output ${save_dir}/pred_novel
