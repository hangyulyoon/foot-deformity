for i in {0..15}
do  
    echo "Training landmark index: $i"
    CCUDA_VISIBLE_DEVICES=2 python train_stage2.py \
    --viewpos frontal \
    --landmark_indices $i \
    --image_size 1024 \
    --dataset_dir /nfs_edlab/ssbae/kaist-2022/foot-deformity-diagnosis/dataset/full_with_report/roi_1024_frontal \
    --unet_encoder_name vgg16 \
    --num_epochs 10 \
    --wandb
done