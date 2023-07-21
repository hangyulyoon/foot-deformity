## Pre-processing for Stage1 (ROI Detection)

```python
python generate_csv_files_for_retinanet.py \
    --dataset_dir=../dataset/full_with_report \
    --viewpos=frontal \ # or lateral
    --frac_ratio=0.8 \
    --roi_size=128

python generate_csv_files_for_retinanet.py \
    --dataset_dir=../dataset/full_with_report \
    --viewpos=frontal \ # or lateral
    --frac_ratio=0.8 \
    --roi_size=1024
```

## Stage1: ROI Detection

```python
CUDA_VISIBLE_DEVICES=0 python train_stage1.py \
--dataset_dir=../dataset/full_with_report/roi_128_frontal \
--viewpos=frontal \ # or lateral
--roi_size=128 \
--model=resnet101 \
--pretrained \
--train_batch_size=4 \
--wandb
```

## Stage2: Landmark Detection

```python
# ceph model
CUDA_VISIBLE_DEVICES=0 python train_stage2.py \
--viewpos frontal \ # or lateral
--landmark_indices 0 \ # index
--image_size 1024 \
--dataset_dir /nfs_edlab/ssbae/kaist-2022/foot-deformity-diagnosis/dataset/full_with_report/roi_1024_frontal \
--unet_encoder_name vgg16 \
--num_epochs 10 \
--wandb

# Flatnet
CUDA_VISIBLE_DEVICES=0 python train_stage2.py \
--viewpos frontal \ # or lateral
--landmark_indices 0 \ # index
--image_size 1024 \
--dataset_dir ../dataset/full_with_report/roi_1024_frontal \
--unet_encoder_name timm-tf_efficientnet_lite4 \
--unet_decoder_attention_type scse \
--num_epochs 10 \
--wandb
```
