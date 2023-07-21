for i in {0..10}
do  
    echo "Training landmark index: $i"
    CUDA_VISIBLE_DEVICES=4 python train_stage2.py \
    --viewpos lateral \
    --landmark_indices $i \
    --image_size 1024 \
    --dataset_dir /nfs_edlab/ssbae/kaist-2022/foot-deformity-diagnosis/dataset/full_with_report/roi_1024_lateral \
    --unet_encoder_name vgg16 \
    --num_epochs 10 \
    --wandb
done