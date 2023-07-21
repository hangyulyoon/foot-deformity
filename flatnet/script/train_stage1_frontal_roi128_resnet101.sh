CUDA_VISIBLE_DEVICES=6,7 python train_stage1.py \
--dataset_dir=/nfs_edlab/ssbae/kaist-2022/foot-deformity-diagnosis/dataset/full_with_report/roi_128_frontal \
--viewpos=frontal \
--roi_size=128 \
--model=resnet101 \
--pretrained \
--train_batch_size=4 \
--wandb