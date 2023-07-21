import os
import cv2
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import pdb
from collections import OrderedDict
import copy 
import pickle
from PIL import Image, ImageDraw
import time

# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from PIL import Image

import torch
import segmentation_models_pytorch as smp


def load_stage1_model(args):
    """
    Load the first stage model.
    """
    try:
        checkpoint_path = args.stage1_ckpt_path
        model = torch.load(checkpoint_path)
        model = model.to(args.device)
        model.training = False
        return model
    except Exception as e:
        print(f"Error loading stage 1 model: {e}")
        return None


def load_stage2_model(args):
    """
    Load the second stage model.
    """
    try:
        # Define the encoder name based on checkpoint directory
        if "timm-tf_efficientnet_lite4" in args.stage2_ckpt_dir:
            encoder_name = "timm-tf_efficientnet_lite4"
        elif "vgg16" in args.stage2_ckpt_dir:
            encoder_name = "vgg16"
        else:
            raise ValueError("Invalid encoder name")

        # Define the decoder attention type based on checkpoint directory
        decoder_attention_type = "scse" if "scse" in args.stage2_ckpt_dir else None

        # Load models for all landmarks
        checkpoint_dir = args.stage2_ckpt_dir
        landmarkid2model = {}
        for landmarkid in os.listdir(checkpoint_dir):
            model = smp.Unet(
                encoder_name=encoder_name,
                decoder_attention_type=decoder_attention_type,
                in_channels=1,
                classes=1,
            )
            weight_path = os.path.join(checkpoint_dir, str(landmarkid), "weight_best.pth")
            weight = torch.load(weight_path)

            if 'module' in [k for k in weight.keys()][0]:
                new_keys = []
                values = weight.values()
                for key in weight:
                    new_key = key[7:]  
                    new_keys.append(new_key)
                new_weight = OrderedDict(list(zip(new_keys, values)))
                model.load_state_dict(new_weight)
            else:
                model.load_state_dict(weight)

            model = model.to(args.device)
            landmarkid2model[int(landmarkid)] = model

        return landmarkid2model
    except Exception as e:
        print(f"Error loading stage 2 model: {e}")
        return None


def Resize(image, x_size=2048):
    y_ori, x_ori = image.shape
    image = cv2.resize(image, (x_size, int((x_size * y_ori) / x_ori)))
    return image, x_size / x_ori

def pil_draw_point(image, point, pred_fill = (255, 0, 0)):
    x, y = point
    draw = ImageDraw.Draw(image)
    radius = 0.5
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill = pred_fill)
    # draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)

    return image


def Pre_Processing(image):
    image = image / 255.0
    image = np.expand_dims(image, 0)
    image = np.expand_dims(image, 0)
    image = image.astype(np.float32)
    image = torch.from_numpy(image)
    image = image.cuda()
    return image


def Get_Center_Points(y_temp):
    xy_args = np.argwhere(y_temp > 0.7) if y_temp.any() else np.argwhere(y_temp == np.max(y_temp))
    y_check, x_check = xy_args.transpose()
    return int(x_check.mean()), int(y_check.mean())


def detect_image(model, image_path):
    image = cv2.imread(image_path)

    if image is None:
        print("Error!")

    # image_orig = image.copy()
    rows, cols, cns = image.shape
    smallest_side = min(rows, cols)

    min_side = 608
    max_side = 1024
    scale = min_side / smallest_side

    largest_side = max(rows, cols)

    if largest_side * scale > max_side:
        scale = max_side / largest_side

    # resize the image with the computed scale
    image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
    #     print('resized image',image.shape)
    rows, cols, cns = image.shape

    pad_w = 32 - rows % 32
    pad_h = 32 - cols % 32

    new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
    new_image[:rows, :cols, :] = image.astype(np.float32)
    image = new_image.astype(np.float32)
    image /= 255
    image -= [0.485, 0.456, 0.406]
    image /= [0.229, 0.224, 0.225]
    image = np.expand_dims(image, 0)
    image = np.transpose(image, (0, 3, 1, 2))

    with torch.no_grad():
        Predict_Point = np.zeros((45, 6), dtype=np.uint32)
        best_scores = [0.0] * 45

        image = torch.from_numpy(image)

        # st = time.time()
        # print(image.shape, image_orig.shape, scale)
        # model = model.cpu()
        # image = image.cpu()
        scores, classification, transformed_anchors = model(image.cuda().float())

        idxs = np.where(scores.cpu() > 0.0)

        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]
            current_class = int(classification[j])

            if scores[j] > best_scores[current_class]:
                best_scores[current_class] = scores[j]
                Predict_Point[current_class][0] = int(scores[j] * 100)
                Predict_Point[current_class][1] = int(bbox[0] / scale)
                Predict_Point[current_class][2] = int(bbox[1] / scale)
                Predict_Point[current_class][3] = int(bbox[2] / scale)
                Predict_Point[current_class][4] = int(bbox[3] / scale)

    return Predict_Point


def main(args):
    start_time = time.time()
    if args.gpu_num:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu_num)

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_OF_LANDMARKS = 16 if args.viewpos == "frontal" else (11 if args.viewpos == "lateral" else ValueError("Invalid viewpos"))

    # Stage 1 inference
    image_paths = pd.read_csv(os.path.join(args.dataset_dir, "annotations.csv"), header=None)[0].unique()
    label_paths = [path.replace("/image/", "/label/").replace(".jpg", ".csv") for path in image_paths]

    stage1_model = load_stage1_model(args)
    stage1_model.eval()

    true_labels, pred_labels = [], []

    for image_path, label_path in tqdm(zip(image_paths, label_paths), total=len(image_paths)):
        pred_info = detect_image(stage1_model, image_path)
        image_orig = cv2.imread(image_path)

        pred_pts = [[int(((pred_info[j][1]) + int(pred_info[j][3])) / 2), int(((pred_info[j][2]) + int(pred_info[j][4])) / 2)] for j in range(NUM_OF_LANDMARKS)]
        true_pts = pd.read_csv(label_path)[["x", "y"]].values

        assert len(pred_pts) == len(true_pts)

        true_labels.append(true_pts)
        pred_labels.append(pred_pts)

    true_labels = np.array(true_labels)  # (N, num_of_landmarks, 2)
    pred_labels = np.array(pred_labels)  # (N, num_of_landmarks, 2)

    print(true_labels.shape, pred_labels.shape)

    pixel_distances = (((true_labels - pred_labels) ** 2).sum(axis=-1)) ** (0.5)
    print("the mean pixel distance is", pixel_distances.mean())

    # Stage 2 inference
    MAX_SIZE = 7000
    shift = patch_size = args.stage2_image_size # 1024
    final_preds = np.zeros((len(image_paths), NUM_OF_LANDMARKS, 2), dtype=np.uint32)

    landmarkid2model = load_stage2_model(args)

    for landmark_idx in range(0, NUM_OF_LANDMARKS):
        landmark_idx = int(landmark_idx)
        model = landmarkid2model[landmark_idx]
        model.eval()

        for j in tqdm(range(len(image_paths))):
            temp_image = cv2.imread(image_paths[j], 0)
            temp_image, ratio = Resize(temp_image)

            start_y = shift
            end_y = shift + temp_image.shape[0]
            end_y = min(end_y, MAX_SIZE)

            start_x = shift
            end_x = shift + temp_image.shape[1]
            end_x = min(end_x, MAX_SIZE)

            temp_image = temp_image[: end_y - start_y, : end_x - start_x]
            padding_image = np.zeros((MAX_SIZE, MAX_SIZE), dtype=np.uint8)
            padding_image[start_y:end_y, start_x:end_x] = temp_image

            left_up = (int((pred_labels[j][landmark_idx][0] * ratio - (patch_size / 2) + shift)), int((pred_labels[j][landmark_idx][1] * ratio - (patch_size / 2) + shift)))
            right_down = (int((pred_labels[j][landmark_idx][0] * ratio + (patch_size / 2) + shift)), int((pred_labels[j][landmark_idx][1] * ratio + (patch_size / 2) + shift)))

            image = padding_image[left_up[1] : right_down[1], left_up[0] : right_down[0]]
            image_shape = copy.deepcopy(image.shape)

            if image.shape != (patch_size, patch_size):
                image = image[:patch_size, :patch_size]
            if image.shape != (patch_size, patch_size):
                print(f'image with inadequate size is created: {image_paths[j]}')
                pdb.set_trace()
                # image = padding_image[right_down[1]-patch_size : right_down[1], right_down[0]-patch_size : right_down[0]]
                # left_up, right_down = list(left_up), list(right_down)
                # add_0 = copy.deepcopy(patch_size - (right_down[1] - left_up[1]))
                # add_1 = copy.deepcopy(patch_size - (right_down[0] - left_up[0]))
                # left_up[0] += add_0
                # right_down[0] += add_1
                # pdb.set_trace()
                # left_up, right_down = tuple(left_up), tuple(right_down)
                # pdb.set_trace()

            output = torch.sigmoid(model(Pre_Processing(image)))
            pr_mask = output.squeeze().cpu().detach().numpy()[:, :].round()

            pr_x_mean, pr_y_mean = Get_Center_Points(pr_mask)
            final_preds[j][landmark_idx][0] = (left_up[0] + pr_x_mean - shift) / ratio
            final_preds[j][landmark_idx][1] = (left_up[1] + pr_y_mean - shift) / ratio

            # ## save the gt & pred landmark with original image
            # temp_image = cv2.imread(image_paths[j], 0)
            # im = Image.fromarray(temp_image).convert('RGB')
            # pt_x, pt_y = (left_up[0] + pr_x_mean - shift) / ratio, (left_up[1] + pr_y_mean - shift) / ratio
            # im = pil_draw_point(im, (pt_x, pt_y))
            # im = pil_draw_point(im, final_preds[j][landmark_idx], pred_fill = (0, 0, 255))
            # im.save("example_file.jpeg")
            # pdb.set_trace()

    end_time = time.time()
    print(f'total_inference_time: {end_time-start_time}')
    pdb.set_trace()

    stage2_pixel_distances = (((true_labels - final_preds) ** 2).sum(axis=-1)) ** (0.5)
    print(f"dataset name: {args.trial} / view: {args.viewpos}")
    print(f"the mean pixel distance is {stage2_pixel_distances.mean():.2f}")
    print(f"the std of pixel distance is {stage2_pixel_distances.std():.2f}")

    ## save the predicted landmarks as the dictionary of {'file_name':'np.array'}
    # image_paths = list of image files 
    print('save the predicted results as .pkl file...')
    prediction_results = dict()
    for j in tqdm(range(len(image_paths))):
        image_key = image_paths[j].split('/')[-1].replace('.jpg', '')
        prediction_results[image_key] = final_preds[j]

    os.makedirs('predictions', exist_ok=True)

    with open(f'predictions/flatnet_{args.viewpos}_{args.trial}.pkl', 'wb') as f: 
        pickle.dump(prediction_results, f)

    print('finish saving')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=str, default="sinchon", help="folder name of the dataset")
    parser.add_argument("--viewpos", type=str, default="frontal")
    parser.add_argument("--stage2_image_size", type=int, default=1024)
    parser.add_argument("--gpu_num", type=int, default=0)
    args = parser.parse_args()

    # NOTE: temporary
    if args.viewpos == "frontal":
        args.dataset_dir = f"/home/server18/hdd/hangyul/foot_deformity/dataset/{args.trial}/roi_128_frontal/"
        args.stage1_ckpt_path = "./models/stage1/frontal_roi-128_model-resnet101_lr-1e-05_pretrained-True_ne-50_bsz-16_seed-42/retinanet_best.pt"
        args.stage2_ckpt_dir = "./models/stage2/frontal_Enc-timm-tf_efficientnet_lite4_Dec-scse_Img-1024_bsz-4_ne-10_seed-42/"
    elif args.viewpos == "lateral":
        args.dataset_dir = f"/home/server18/hdd/hangyul/foot_deformity/dataset/{args.trial}/roi_128_lateral"
        args.stage1_ckpt_path = "./models/stage1/lateral_roi-128_model-resnet101_lr-1e-05_pretrained-True_ne-50_bsz-16_seed-42/retinanet_best.pt"
        args.stage2_ckpt_dir = "./models/stage2/lateral_Enc-timm-tf_efficientnet_lite4_Dec-scse_Img-1024_bsz-4_ne-10_seed-42/"
    else:
        raise ValueError()

    main(args)
