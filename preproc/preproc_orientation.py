import os
import re
import cv2
import shutil
from tqdm import tqdm
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# custom pkgs
from correction import CORRECTION_DICT
from rotation import ROTATION_DICT
from flipping import FLIPPING_DICT


NAMERULE_OF_VIEWPOS = {
    "AP": "A",
    "LT": "L",
    "RT": "R",
    "A(R)": "AR",
    "A(L)": "AL",
}


def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text


def build_image_meta(image_dir):

    _meta_data = {"image_dir": image_dir, "image_name": os.listdir(image_dir)}
    meta_data = pd.DataFrame(_meta_data)

    meta_data["image_name_split"] = meta_data["image_name"].apply(lambda x: x.split(".")[0])
    meta_data["patient_id"] = meta_data["image_name_split"].apply(lambda x: re.search(r"\d+", x).group())
    meta_data["patient_id"] = meta_data["patient_id"].astype(int)
    meta_data["view_pos"] = meta_data["image_name_split"].apply(lambda x: re.search(r"[a-zA-Z(a-z)\[A-Z\]]+", x).group())
    print(meta_data["view_pos"].unique())
    del meta_data["image_name_split"]

    meta_data["seq_id"] = meta_data["image_name"].apply(lambda x: x.split(".")[1])
    meta_data["ser_id"] = meta_data["image_name"].apply(lambda x: x.split(".")[2])
    meta_data["image_id"] = meta_data["image_name"].apply(lambda x: x.split(".")[3])

    meta_data["image_path"] = meta_data["image_dir"] + "/" + meta_data["image_name"]
    meta_data["landmark_id"] = meta_data["patient_id"].astype(str) + meta_data["view_pos"]

    return meta_data


def main(args):

    # timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # set image directory
    raw_image_dir, new_image_dir = args.raw_image_dir, args.new_image_dir
    new_image_dir = os.path.join(new_image_dir, timestamp)
    if not os.path.exists(raw_image_dir):
        raise ValueError(f"{raw_image_dir} (raw image dir) does not exist.")
    if os.path.exists(new_image_dir):
        raise ValueError(f"{new_image_dir} (new image dir) already exists. Please remove it.")
    os.makedirs(new_image_dir, exist_ok=True)

    # get file list
    raw_image_names = os.listdir(raw_image_dir)

    # build new image name
    raw2new_dict = {}
    for raw_image_name in tqdm(raw_image_names):

        # init
        new_image_name = raw_image_name

        # NOTE: fix comma issue
        if raw_image_name == "8256388R..Seq1.Ser2098.Img1.jpg":
            new_image_name = "8256388R.Seq1.Ser2098.Img1.jpg"

        # NOTE: remove duplicated issue
        if "(1)" in raw_image_name:
            continue

        # assertion check: file name must be 5 parts
        assert len(new_image_name.split(".")) == 5
        x, seq_id, ser_id, image_id, ext = new_image_name.split(".")

        # 1st part consists of patient id and view position
        patient_id = re.search(r"\d+", x).group()
        viewpos = re.search(r"[a-zA-Z(a-z)\[A-Z\]]+", x).group()
        viewpos = viewpos.upper()

        # rename view position
        viewpos = replace_all(viewpos, NAMERULE_OF_VIEWPOS)

        # gather names
        new_image_name = f"{patient_id}{viewpos}.{seq_id}.{ser_id}.{image_id}.jpg"

        # correct image name
        if raw_image_name in CORRECTION_DICT.keys():
            new_image_name = CORRECTION_DICT[raw_image_name]

        # save
        raw2new_dict[raw_image_name] = new_image_name

    # copy (raw to new directory)
    for raw_image_name, new_image_name in tqdm(raw2new_dict.items()):

        raw_image_path = os.path.join(raw_image_dir, raw_image_name)
        new_image = cv2.imread(raw_image_path, cv2.IMREAD_GRAYSCALE)

        # rotation
        rotation = 0
        if new_image_name in ROTATION_DICT.keys():
            rotation = ROTATION_DICT[new_image_name]
        if rotation == 90:
            new_image = cv2.rotate(new_image, cv2.ROTATE_90_CLOCKWISE)
        elif rotation == 180:
            new_image = cv2.rotate(new_image, cv2.ROTATE_180)
        elif rotation == -90:
            new_image = cv2.rotate(new_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        elif rotation == 0:
            pass
        else:
            raise ValueError()

        # flipping
        flipping = False
        if new_image_name in FLIPPING_DICT.keys():
            flipping = FLIPPING_DICT[new_image_name]
        if flipping:
            new_image = cv2.flip(new_image, 1)

        new_image_path = os.path.join(new_image_dir, new_image_name)
        cv2.imwrite(new_image_path, new_image)

    # build meta data
    meta = build_image_meta(image_dir=new_image_dir)

    # debug check
    if args.debug:
        print("debug check")
        # debug_dir = args.debug_dir
        debug_dir = os.path.join(args.debug_dir, timestamp)
        for record in tqdm(meta.to_dict("records")):
            image_path = record["image_path"]
            image_name = record["image_name"]
            view_pos = record["view_pos"]
            os.makedirs(os.path.join(debug_dir, view_pos), exist_ok=True)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite(os.path.join(debug_dir, view_pos, image_name), image)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.debug = False
    args.debug_dir = "debug"
    # args.raw_image_dir = "./raw_image/Raw images (영상판독 O)"
    # args.new_image_dir = "./new_image/Raw images (영상판독 O)"

    args.raw_image_dir = "/home/server18/hangyul/gangnam_image"
    args.new_image_dir = "/home/server18/hangyul/gangnam_image_new"

    main(args)
