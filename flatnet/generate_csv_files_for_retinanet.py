import os
import re
import cv2
import pandas as pd


def generate_annotations(image_path, label_path, roi_size, debug=False):
    """
    final output format:
    path/to/image.jpg,x1,y1,x2,y2,class_name
    """
    # read image and get width and height
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    # get a label
    labels = pd.read_csv(label_path)
    labels = labels.sort_values(by="index")
    labels["index"] = [idx for idx in range(1, len(labels) + 1)]
    labels = labels.set_index("index").to_dict("index")

    annotations = []
    is_ap_view = True if "A" in image_path else False

    # number_of_landmarks = 14
    # assert len(labels) == number_of_landmarks, [len(labels), label_path]

    if debug:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    for label_idx in labels.keys():
        label = labels[label_idx]

        # make roi from a label
        center_x, center_y = int(label["x"]), int(label["y"])

        if debug:
            # check image save with landmarks
            cv2.circle(
                img=image,
                center=(center_x, center_y),
                radius=3,
                color=(0, 0, 255),
                thickness=-1,
            )

        # TODO: minus values are not allowed?
        x1 = center_x - roi_size / 2
        y1 = center_y - roi_size / 2
        x2 = center_x + roi_size / 2
        y2 = center_y + roi_size / 2

        # try:
        #     label_idx = label_idx.replace("F-", "")
        #     label_idx = str(label_idx)
        # except:
        #     pass

        class_name = f"A{label_idx}" if is_ap_view else f"L{label_idx}"
        annotation = {
            "image_path": image_path,
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "class_name": class_name,
        }
        annotations.append(annotation)

    if debug:
        debug_save_path = f"debug_{os.path.basename(image_path)}"
        cv2.imwrite(debug_save_path, image)

    return annotations


def main(args):
    # read meta data
    metadata = pd.read_csv(os.path.join(args.dataset_dir, "metadata.csv")) # composed of 2 columns -> image_name & label_name
    records = metadata.to_dict("records") # change the df into the list with form of [{'col1': 1, 'col2': 2}, {'col1': 3, 'col2': 4}]

    # build annotations csv
    annotations_csv = []
    for record in records:

        image_name = record["image_name"]
        label_name = record["label_name"]

        image_path = os.path.join(args.dataset_dir, "image", image_name)
        label_path = os.path.join(args.dataset_dir, "label", label_name)

        # NOTE:
        image_path = os.path.abspath(image_path) 
        label_path = os.path.abspath(label_path) 

        # viewpos == A, AL, AR, L, R
        viewpos = re.sub(r"[^A-Z]", "", label_name.split(".")[0])

        if args.viewpos == "lateral":
            if viewpos.startswith("L") or viewpos.startswith("R"):
                print(viewpos, image_path, label_path)
                annotations = generate_annotations(image_path, label_path, roi_size=args.roi_size)
                annotations_csv.extend(annotations)
        elif args.viewpos == "frontal":
            if viewpos.startswith("A"):
                print(viewpos, image_path, label_path)
                annotations = generate_annotations(image_path, label_path, roi_size=args.roi_size)
                annotations_csv.extend(annotations)
        elif args.viewpos == "both":
            raise NotImplementedError()
        else:
            raise ValueError()

    annotations_csv = pd.DataFrame(annotations_csv)

    # split and save train/val (group by image path)
    # TODO: should be modified
    total_image_paths = annotations_csv["image_path"].unique().tolist()

    # build classes csv
    if args.viewpos == "frontal":
        classes = [f"A{idx+1}" for idx in range(16)]
    elif args.viewpos == "lateral":
        classes = [f"L{idx+1}" for idx in range(11)]
    elif args.viewpos == "both":
        raise NotImplementedError()
        # classes_for_ap = [f"A{idx+1}" for idx in range(16)]
        # classes_for_lateral = [f"L{idx+1}" for idx in range(11)]
        # classes = classes_for_ap + classes_for_lateral
    else:
        raise ValueError(f"Invalid viewpos: {args.viewpos}")
    classes_csv = pd.DataFrame({"class_name": classes, "id": range(len(classes))})

    # define save dir
    save_dir = os.path.join(args.dataset_dir, f"roi_{args.roi_size}_{args.viewpos}")
    os.makedirs(save_dir, exist_ok=True)

    # save annotations csv
    annotations_csv.to_csv(
        os.path.join(save_dir, "annotations.csv"),
        index=False,
        header=False,
    )

    # save classes csv (remove column and index)
    classes_csv.to_csv(
        os.path.join(save_dir, "classes.csv"),
        index=False,
        header=False,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="/home/server18/hdd/hangyul/foot_deformity/dataset/sinchon")
    parser.add_argument("--viewpos", type=str, choices=["frontal", "lateral", "both"], default="lateral")
    # parser.add_argument("--frac_ratio", type=float, default=1.0)
    parser.add_argument("--roi_size", type=int, choices=[128, 512, 1024], default=512)
    args = parser.parse_args()

    main(args)
