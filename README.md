# foot-deformity-diagnosis


## Reference
- Ryu, Seung Min, et al. "Automated landmark identification for diagnosis of the deformity using a cascade convolutional neural network (FlatNet) on weight-bearing lateral radiographs of the foot." Computers in Biology and Medicine 148 (2022) [[paper]](https://dl.acm.org/doi/abs/10.1016/j.compbiomed.2022.105914) [[code]](https://github.com/kevinkwshin/FlatNet)
- Lan, Xing, Qinghao Hu, and Jian Cheng. "Revisting quantization error in face alignment." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021. [[paper]](https://openaccess.thecvf.com/content/ICCV2021W/MFR/papers/Lan_Revisting_Quantization_Error_in_Face_Alignment_ICCVW_2021_paper.pdf)
- Lan, Xing, Qinghao Hu, and Jian Cheng. "HIH: Towards more accurate face alignment via heatmap in heatmap." arXiv preprint arXiv:2104.03100 (2021). [[paper]](https://arxiv.org/abs/2104.03100) [[code]](https://github.com/starhiking/HeatmapInHeatmap)


## Usage 
* Upload data and label to **./dataset/XXX/raw_image**, **./dataset/XXX/raw_label**. 
* Run **python .\dataset\preprocess_dataset.py --dataset_dir ./dataset/XXX**. Processed images and labels are saved to **./dataset/XXX/image** and **./dataset/XXX/label** 
* (Optional) Run **python .\dataset\dataloader.py --dataset_dir ./dataset/XXX** to check data and augmentation integrity. Debug images are saved to **./dataset/XXX/augmentation_samples** 
* (Optional) Run **python main_HIH.py --dataset_dir ./dataset/XXX --fast_pass** to check everything goes fine. 
* **Run python main_HIH.py --dataset_dir ./dataset/XXX**, or **python main_HIH.py --dataset_dir ./dataset/XXX --top_view** for AP view. Checkpoints are saved to **./checkpoint/{exp_name}_best.pth**. Training logs and sample visualizations are saved to **./training_logs/{current_time}**. 
* Run **python inference.py --data_dir ./dataset/XXX --resume ./checkpoint/{exp_name}_best.pth**. Inference code is not fully verified yet, so it can be erroneous. 
* For the implementation of FlatNet, please refer to the **./foot-deformity/flatnet/README.md** file. 
