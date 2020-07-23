CUDA_VISIBLE_DEVICES=3 python inference_on_det.py \
--root /home/xiao/Datasets/Pascal3D \
--det /home/xiao/Datasets/Pascal3D/results \
--save /home/xiao/Datasets/Pascal3D/meta_results \
--model exps/ObjectNet3D_pc/ObjectNet3D_pc_TDID_RandModel_shot10_1/checkpoint.pth \
--att exps/ObjectNet3D_pc/ObjectNet3D_pc_TDID_RandModel_shot10_1/mean_class_attentions.pkl