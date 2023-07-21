for i in {0..15}
do  
    echo "Training landmark index: $i"
    CUDA_VISIBLE_DEVICES=3 python train_stage2.py \
    --viewpos frontal \
    --landmark_indices $i \
    --image_size 1024 \
    --dataset_dir /nfs_edlab/ssbae/kaist-2022/foot-deformity-diagnosis/dataset/full_with_report/roi_1024_frontal \
    --unet_encoder_name timm-tf_efficientnet_lite4 \
    --unet_decoder_attention_type scse \
    --num_epochs 10 \
    --wandb
done