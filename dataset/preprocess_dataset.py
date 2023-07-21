import os
import cv2
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import re
import pdb

from tqdm import tqdm

def read_label_file(label_path):
    # label .fcsv file
    df = pd.read_csv(
        label_path,
        sep=",",
        skiprows=3,
        names=["id", "x", "y", "z", "ow", "ox", "oy", "oz", "vis", "sel", "lock", "label", "desc", "associatedNodeID", "temp", "temp2"],
    )

    df = df[["id", "x", "y"]].set_index("id")

    # if pd.unique(df.index).shape[0] != 11 and pd.unique(df.index).shape[0] != 16:
    #     print(f'index duplication error : {label_path}')

    try:
        labels = df.to_dict("index") # 1~11 (AL/AR) or 1~16 (L/R)
    except:
        print(f'fscv file index error : {label_path}')

    return labels

def crop_image_by_contour(image, labels, debug=False, debug_save_path=None):

    # binarize image
    ret, thresh = cv2.threshold(image, 0, 255, 0)

    # get contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    areas = np.array([cv2.contourArea(c) for c in contours])

    # get the largest contour
    max_index = np.argmax(areas)
    cnt = contours[max_index]
    x, y, w, h = cv2.boundingRect(cnt)
    x1, x2 = x, x + w
    y1, y2 = y, y + h

    # NOTE: merge all countours if the contour is larger than 5% of total pixels (for now, AP)
    tot_pixels = len(image.flatten())
    n_areas_pct5 = sum(areas >= tot_pixels * 0.05)

    if n_areas_pct5 > 1:
        max_indices = np.argsort(areas)[-n_areas_pct5:]
        for idx in max_indices:
            cnt = contours[idx]
            x, y, w, h = cv2.boundingRect(cnt)
            x1, y1 = min(x1, x), min(y1, y)
            x2, y2 = max(x2, x + w), max(y2, y + h)
            
    # update by labels + margin
    margin = 100
    label_min_x, label_min_y, label_max_x, label_max_y = get_minmax_coordinates_of_labels(labels)
    x1, y1 = min(x1, label_min_x-margin), min(y1, label_min_y-margin)
    x2, y2 = max(x2, label_max_x+margin), max(y2, label_max_y+margin)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, image.shape[1]), min(y2, image.shape[0])

    # crop image
    cropped_image = image[y1:y2, x1:x2]
    offsets = [x1, y1, x2, y2]

    if debug:
        fig, axes = plt.subplots(4, 1, figsize=(8, 12))  # , gridspec_kw={'height_ratios':[10,10,3,10]})
        axes[0].imshow(image, cmap="gray")
        axes[1].imshow(thresh, cmap="gray")
        axes[2].imshow(cropped_image, cmap="gray")
        axes[3].plot(areas)
        fig.set_facecolor("white")
        plt.tight_layout()
        plt.subplots_adjust()
        plt.savefig(debug_save_path)
        plt.clf()

    return cropped_image, offsets


def get_minmax_coordinates_of_labels(labels):
    # get coordinates of labels
    label_min_x = int(min([label["x"] for label in labels.values()]))
    label_min_y = int(min([label["y"] for label in labels.values()]))
    label_max_x = int(max([label["x"] for label in labels.values()]))
    label_max_y = int(max([label["y"] for label in labels.values()]))
    return (label_min_x, label_min_y, label_max_x, label_max_y)


def update_labels_by_offsets(labels, offsets):
    new_labels = {}
    for label_id, label_coord in labels.items():
        new_labels[label_id] = {
            "x": label_coord["x"] - offsets[0],
            "y": label_coord["y"] - offsets[1],
        }
    return new_labels


def flip_labels(labels, width):
    new_labels = {}
    for label_id, label_coord in labels.items():
        new_labels[label_id] = {
            "x": width - label_coord["x"],
            "y": label_coord["y"],
        }
    return new_labels


def split_two_foots(image, labels, viewpos):

    edge_cut = True
    if edge_cut:
        offset = 100
    else:
        offset = 0

    _, width = image[:, offset:-offset].shape
    width_indices = range(width)
    width_values = image[:, offset:-offset].mean(axis=0)

    # find peaks
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(width_values, distance=width / 5)
    peaks = peaks[np.argsort(-width_values[peaks])[:2]]  # among them, choose two best peaks
    peaks_values = width_values[peaks]

    # save left / right
    peak_l, peak_r = int(peaks[0]), int(peaks[1])
    if peak_l > peak_r:
        peak_l, peak_r = peak_r, peak_l

    min_value = width_values[peak_l:peak_r].min()
    min_indices = np.where(width_values[peak_l:peak_r] <= min_value + 10)[0]
    min_indices = sorted(min_indices)

    min_l = offset + peak_l + min_indices[0]
    min_r = offset + peak_l + min_indices[-1]
    
    if viewpos in ["L", "AL"]:
        x1, y1 = 0, 0
        x2, y2 = min_r, image.shape[0]
    elif viewpos in ["R", "AR"]:
        x1, y1 = min_l, 0
        x2, y2 = image.shape[1], image.shape[0]
    else:
        raise ValueError("viewpos should be L or R")
    
    # update by labels + margin
    margin = 25
    margin2 = 25
    label_min_x, label_min_y, label_max_x, label_max_y = get_minmax_coordinates_of_labels(labels)
    x1, y1 = min(x1+margin2, label_min_x-margin), min(y1, label_min_y-margin)
    x2, y2 = max(x2-margin2, label_max_x+margin), max(y2, label_max_y+margin)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, image.shape[1]), min(y2, image.shape[0])
    
    splitted_image = image[y1:y2, x1:x2]
    offsets = [x1, y1, x2, y2]
        
    return splitted_image, offsets


def generate_metadata_from_raw_dataset(raw_image_dir, raw_label_dir):
    # build metadata csv
    metadata = []
    for label_name in os.listdir(raw_label_dir):

        # get viewpos from label name
        label_id, _ = label_name.split(".")  # 760991AL
        patient_id = re.sub(r"[^0-9]", "", label_id)  # 760991
        viewpos_for_label = re.sub(r"[^A-Z]", "", label_id)  # AL
        try: viewpos_for_label in ["AR", "AL", "LR", "L", "R"], viewpos_for_label 
        except: print((f"viewpos is not valid: {viewpos_for_label}, label_id: {label_id}"))

        # find the proper suffix (i.e., viewpos) for image
        if viewpos_for_label == "AR":
            viewpos_for_image = ["A", "AR"]
        elif viewpos_for_label == "AL":
            viewpos_for_image = ["A", "AL"]
        elif viewpos_for_label == "L":
            viewpos_for_image = ["L", "LR", "RL"]
        elif viewpos_for_label == "R":
            viewpos_for_image = ["R", "LR", "RL"]
        else:
            raise ValueError(f"viewpos is not valid: {viewpos_for_label}, label_id: {label_id}, label_file_name: {label_name}")

        # find the corresponding image name
        image_name = []
        for image_name_ in os.listdir(raw_image_dir):
            for viewpos in viewpos_for_image:
                if f"{patient_id}{viewpos}" == image_name_.split('.')[0] :
                    image_name.append(image_name_)
        if len(set(image_name)) == 0 : 
            print(f'Image not exists for {label_name}')
            continue 
        elif len(set(image_name)) >= 2 : 
            print(f'Multiple images ({image_name}) for single label ({label_name}). The first one is selected.')
            
        image_name = image_name[0]

        # append to metadata
        metadata.append(
            {
                "image_name": image_name,
                "label_name": label_name,
            }
        )

    # save metadata csv
    metadata = pd.DataFrame(metadata)
    return metadata

def split_two_foots_gangnam(image, labels, viewpos):
    img_x, img_y = image.shape[1], image.shape[0]
    if viewpos in ["L", "AL"]:
        x1, y1 = 0, 0
        x2, y2 = int(img_x * 0.55), image.shape[0]
    elif viewpos in ["R", "AR"]:
        x1, y1 = int(img_x * 0.55), 0
        x2, y2 = image.shape[1], image.shape[0]
    else:
        raise ValueError("viewpos should be L or R")
    
    # update by labels + margin
    margin = 100
    label_min_x, label_min_y, label_max_x, label_max_y = get_minmax_coordinates_of_labels(labels)
    x1, y1 = min(x1, label_min_x-margin), min(y1, label_min_y-margin)
    x2, y2 = max(x2, label_max_x+margin), max(y2, label_max_y+margin)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, image.shape[1]), min(y2, image.shape[0])
    
    splitted_image = image[y1:y2, x1:x2]
    offsets = [x1, y1, x2, y2]
        
    return splitted_image, offsets

def main(args):
    
    # raw
    image_dir = os.path.join(args.dataset_dir, "raw_image")
    label_dir = os.path.join(args.dataset_dir, "raw_label")
    raw_metadata = generate_metadata_from_raw_dataset(raw_image_dir=image_dir, raw_label_dir=label_dir)
    print("number of images:", len(os.listdir(image_dir)))
    print("number of labels:", len(os.listdir(label_dir)))
    
    # new
    new_image_dir = os.path.join(args.dataset_dir, "image")
    new_label_dir = os.path.join(args.dataset_dir, "label")
    os.makedirs(new_image_dir, exist_ok=True)
    os.makedirs(new_label_dir, exist_ok=True)
    
    records = raw_metadata.to_dict("records")
    for record in tqdm(records):
        
        image_name = record["image_name"]
        label_name = record["label_name"]
        
        image_path = os.path.join(args.dataset_dir, "raw_image", image_name)
        label_path = os.path.join(args.dataset_dir, "raw_label", label_name)
        
        image_id = image_name.split(".")[0]
        label_id = label_name.split(".")[0]
        viewpos_for_label = re.sub(r"[^A-Z]", "", label_id)
        viewpos_for_image = re.sub(r"[^A-Z]", "", image_id)
        
        # read the image and labels
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        labels = read_label_file(label_path)
        
        # crop the image and update labels
        new_image, offsets = crop_image_by_contour(image=image, labels=labels)
        labels = update_labels_by_offsets(labels=labels, offsets=offsets)
        
        # mark the `has_two_foots` flag
        has_two_foots = False
        if viewpos_for_image in ["A", "LR", "RL"]:
            has_two_foots = True
        elif viewpos_for_image in ["AL", "AR", "L", "R"]:
            pass
        else:
            raise ValueError("viewpos_for_image is not valid")
        
        # if the image has two foots, then split the image into two images
        if has_two_foots:
            if viewpos_for_image == "RL":
                # for RL, flip the image and labels (=> LR)
                new_image = cv2.flip(new_image, 1)
                labels = flip_labels(labels=labels, width=new_image.shape[1])
            
            # split the image
            new_image, offsets2 = split_two_foots(image=new_image, labels=labels, viewpos=viewpos_for_label)
            labels = update_labels_by_offsets(labels=labels, offsets=offsets2)
        
        # save image
        new_image_name = label_name.replace(".fcsv", ".jpg")
        new_image_path = os.path.join(new_image_dir, new_image_name)
        cv2.imwrite(new_image_path, new_image)
            
        # save labels
        new_label_name = label_name.replace(".fcsv", ".csv")
        new_label_path = os.path.join(new_label_dir, new_label_name)
        new_labels = pd.DataFrame(labels).T.reset_index()
        new_labels.to_csv(new_label_path, index=False)
            
        # NOTE: debugging mode
        # draw the cropped image with labels (i.e., landmarks)
        if args.debug:
            # set debug dir
            debug_dir = os.path.join(args.dataset_dir, "debug")
            os.makedirs(debug_dir, exist_ok=True)
            
            # draw labels (i.e., landmarks)
            for label_id, label_coord in labels.items():
                center_x = int(label_coord["x"])
                center_y  = int(label_coord["y"])
                cv2.putText(
                    img=new_image, 
                    text=f"{label_id}", 
                    org=(center_x, center_y), 
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0, 0, 255),
                    # lineType=cv2.LINE_AA,
                )
                cv2.circle(
                    img=new_image, 
                    center=(center_x, center_y), 
                    radius=3,  
                    color=(0, 0, 255),
                    thickness=-1,
                )
                
            # save image with labels
            debug_save_path = os.path.join(debug_dir, new_image_name)
            cv2.imwrite(debug_save_path, new_image)
            
    # generate new metadata and save it
    new_image_names = [new_image_name for new_image_name in os.listdir(new_image_dir)]
    new_label_names = list(map(lambda x: x.replace(".jpg", ".csv"), new_image_names))
    new_metadata = pd.DataFrame({"image_name":new_image_names, "label_name": new_label_names})
    new_metadata.to_csv(os.path.join(args.dataset_dir, "metadata.csv"), index=False)
    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, default="./dataset/final")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    
    main(args)
