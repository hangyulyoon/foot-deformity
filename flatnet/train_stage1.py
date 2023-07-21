# !pip install pycocotools --user
import os
import argparse
import collections
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageDraw

# retinanet pkgs
from stage1 import model
from stage1 import csv_eval
from tqdm import tqdm

# from retinanet import coco_eval
from stage1.dataloader import (
    # CocoDataset,
    CSVDataset,
    collater,
    Resizer,
    AspectRatioBasedSampler,
    Augmenter,
    Normalizer,
    UnNormalizer
)

import pdb
import time

MODEL_CLASSES = {
    "resnet18": model.resnet18,
    "resnet34": model.resnet34,
    "resnet50": model.resnet50,
    "resnet101": model.resnet101,
    "resnet152": model.resnet152,
}


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):

    if args.model_save_path is None:
        # from datetime import datetime

        # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_tag = "_".join(
            [
                f"{args.viewpos}",
                f"roi-{args.roi_size}",
                f"model-{args.model}",
                f"lr-{args.lr}",
                f"pretrained-{args.pretrained}",
                f"ne-{args.num_epochs}",
                f"bsz-{args.train_batch_size}",
                f"seed-{args.seed}",
            ]
        )
        model_save_path = os.path.join("models", "stage1", model_tag)
    else:
        model_save_path = args.model_save_path

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    if args.wandb:
        import wandb

        wandb.init(
            project=args.wandb_project,
            config=args,
            name=model_tag,
        )

    # set seed
    set_seed(seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    train_file_path = os.path.join(args.dataset_dir, f"annotations.csv")
    valid_file_path = os.path.join(args.valid_dir, f"annotations.csv")
    class_file_path_train = os.path.join(args.dataset_dir, f"classes.csv")
    class_file_path_valid = os.path.join(args.valid_dir, f"classes.csv")

    # dataloader - train

    dataset_train = CSVDataset(train_file=train_file_path, class_list=class_file_path_train, transform=transforms.Compose([Augmenter(), Normalizer(), Resizer()]))
    dataset_train_ = CSVDataset(train_file=train_file_path, class_list=class_file_path_train, transform=transforms.Compose([Normalizer(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=args.train_batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=0, collate_fn=collater, batch_sampler=sampler)

    # dataloader - valid
    dataset_val = CSVDataset(train_file=valid_file_path, class_list=class_file_path_valid, transform=transforms.Compose([Normalizer(), Resizer()]))
    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=args.valid_batch_size, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=0, collate_fn=collater, batch_sampler=sampler_val)

    # ex_ = dataset_train[0]['img'].numpy()

    def scaling_function(image):
        image_mean = np.array([[[0.485, 0.456, 0.406]]])
        image_std = np.array([[[0.229, 0.224, 0.225]]])

        return (image.astype(np.float32)-image_mean)/image_std
    
    def descaling_function(image):
        image_mean = np.array([[[0.485, 0.456, 0.406]]])
        image_std = np.array([[[0.229, 0.224, 0.225]]])

        return (image.astype(np.float32)*image_std)+image_mean
    
    def pil_draw_point(image, point):
        x, y = point
        draw = ImageDraw.Draw(image)
        radius = 2
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 0, 0))

        return image
    
    # samples = [dataset_train[i] for i in range(20)]
    # samples_ = [dataset_train_[i] for i in range(20)]
    # img_list = [sample['img'] for sample in samples]
    # img_list_ = [sample['img'] for sample in samples_]
    # annot_list = [sample['annot'] for sample in samples]
    # annot_list_ = [sample['annot'] for sample in samples_]

    # os.makedirs('./example_figures', exist_ok=True)
    # for i in range(len(img_list)):
    #     img = Image.fromarray((descaling_function(img_list[i].numpy()) * 255.).astype('uint8'))
    #     img_ = Image.fromarray((descaling_function(img_list_[i].numpy()) * 255.).astype('uint8'))
    #     pdb.set_trace()
    #     for landmark_num in range(annot_list[i].shape[0]):
    #         img = pil_draw_point(img, (annot_list[i][landmark_num, :2]+annot_list[i][landmark_num, 2:4])/2)
    #     for landmark_num in range(annot_list_[i].shape[0]):
    #         img_ = pil_draw_point(img_, (annot_list_[i][landmark_num, :2]+annot_list_[i][landmark_num, 2:4])/2)
    #     img.save(f"./example_figures/img{i}.jpeg")
    #     img_.save(f"./example_figures/img_{i}.jpeg")

    # pdb.set_trace()

    # backbone model
    retinanet = MODEL_CLASSES[args.model](
        num_classes=dataset_train.num_classes(),
        pretrained=True if args.pretrained else False,
    )

    # use_gpu = USE_GPU
    retinanet = retinanet.to(device)
    retinanet = torch.nn.DataParallel(retinanet).to(device)

    retinanet.training = True
    optimizer = optim.AdamW(retinanet.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()
    print("Num training images: {}".format(len(dataset_train)))

    best_val_mean_map = 0

    for epoch_num in range(args.num_epochs):
        start_time = time.time()
        retinanet.train()
        retinanet.module.freeze_bn()
        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                classification_loss, regression_loss = retinanet([data["img"].to(device).float(), data["annot"]])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                # wandb logging
                if args.wandb:
                    wandb.log(
                        {
                            "train/loss": loss,
                            "train/classification_loss": classification_loss,
                            "train/regression_loss": regression_loss,
                        }
                    )

                if bool(loss == 0):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                optimizer.step()
                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))

                print(
                    "Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}".format(
                        epoch_num, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)
                    )
                )

                del classification_loss
                del regression_loss

            except Exception as e:
                print(e)
                continue

        end_time = time.time()
        print(f'time: {end_time-start_time}')
        pdb.set_trace()
        print("Evaluating dataset")
        # retinanet.eval()
        # total_val_loss = 0
        # total_val_classification_loss = 0
        # total_val_regression_loss = 0
        # for iter_num, data in enumerate(dataloader_val):
        #     val_classification_loss, val_regression_loss = retinanet([data["img"].to(device).float(), data["annot"]])
        #     val_classification_loss = val_classification_loss.mean().cpu()
        #     val_regression_loss = val_regression_loss.mean().cpu()
        #     val_loss = val_classification_loss + val_regression_loss

        #     total_val_classification_loss += val_classification_loss
        #     total_val_regression_loss += val_regression_loss
        #     total_val_loss += val_loss
        # if args.wandb:
        #     wandb.log(
        #         {
        #             "valid/loss": total_val_loss,
        #             "valid/classification_loss": total_val_classification_loss,
        #             "valid/regression_loss": total_val_regression_loss,
        #         }
        #     )

        val_mean_map = 0
        mAP = csv_eval.evaluate(dataset_val, retinanet)

        end_time_2 = time.time()
        print(f'time: {end_time_2-end_time}')
        pdb.set_trace()

        # for i in range(25):
        #     mean_map += mAP[i][0]

        # for i in range(11):
        #     val_mean_map += mAP[i][0]
        val_mean_map = np.mean([v[0] for v in mAP.values()])

        scheduler.step(np.mean(epoch_loss))

        if val_mean_map > best_val_mean_map:

            best_val_mean_map = val_mean_map

            # save per epoch if better than previous
            # torch.save(retinanet.module, os.path.join(model_save_path, f"retinanet_{epoch_num}.pt"))

            # save the best model
            torch.save(retinanet.module, os.path.join(model_save_path, f"retinanet_best.pt"))

            print("#" * 60)
            print("model save!!!!!!!_{}.pt".format(best_val_mean_map))
            print("#" * 60)

        if args.wandb:
            wandb.log(
                {
                    "valid/mean_map": val_mean_map,
                    "valid/best_mean_map": best_val_mean_map,
                }
            )

    # save the final model
    retinanet.eval()
    torch.save(retinanet, os.path.join(model_save_path, f"retinanet_final.pt"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple training script for training a RetinaNet network.")

    # core
    parser.add_argument("--viewpos", help="viewpos", default="frontal", choices=["frontal", "lateral"])
    parser.add_argument("--roi_size", help="roi_size", default=128, type=int)

    # directory
    parser.add_argument("--dataset_dir", help="Dataset directory path", required=True)
    parser.add_argument("--valid_dir", help="Dataset directory path for validation", required=True)
    parser.add_argument("--model_save_path", help="Model save path")

    # model config
    parser.add_argument("--model", help="Model type", default="resnet101", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"])
    parser.add_argument("--pretrained", help="Use pretrained model", action="store_true")

    # train config
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr", help="Learning rate", default=1e-5, type=float)
    parser.add_argument("--num_epochs", help="Number of epochs", default=50, type=int)
    parser.add_argument("--train_batch_size", help="Batch size", default=1, type=int)
    parser.add_argument("--valid_batch_size", help="Batch size", default=1, type=int)

    # wandb config
    parser.add_argument("--wandb", help="Use wandb", action="store_true")
    parser.add_argument("--wandb_project", help="Wandb project name", default="flatnet_stage1")

    args = parser.parse_args()

    main(args)
