import os
import copy
import numpy as np
import pandas as pd


import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# custom
from stage2.transform import *
from stage2.progressbar import Bar
from stage2.data_generator import *
from stage2.Unet import *
import time
import pdb


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    ## set the gpu number 

    # if args.image_size == 1024:
    #     assert args.train_batch_size == 4, "batch size should be 4"
    # elif args.image_size == 512:
    #     assert args.train_batch_size == 16, "batch size should be 16"
    # else:
    #     raise NotImplementedError

    if args.viewpos == "frontal":
        NUM_OF_LANDMARKS = 16
    elif args.viewpos == "lateral":
        NUM_OF_LANDMARKS = 11
    else:
        raise ValueError

    model_tag = "_".join(
        [
            args.viewpos,
            f"Enc-{args.unet_encoder_name}",
            f"Dec-{args.unet_decoder_attention_type}",
            f"Img-{args.image_size}",
            f"bsz-{args.train_batch_size}",
            f"ne-{args.num_epochs}",
            f"seed-{args.seed}",
        ]
    )
    if args.model_save_path is None:
        from datetime import datetime

        # timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_save_path = os.path.join("models", "stage2", model_tag)
    else:
        model_save_path = args.model_save_path

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    # set seed
    set_seed(seed=args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    x_train = pd.read_csv(os.path.join(args.dataset_dir, "annotations.csv"), header=None)[0].unique()
    y_train = [path.replace("/image/", "/label/").replace(".jpg", ".csv") for path in x_train]
    x_val = pd.read_csv(os.path.join(args.valid_dir, "annotations.csv"), header=None)[0].unique()
    y_val = [path.replace("/image/", "/label/").replace(".jpg", ".csv") for path in x_val]
    print(f"train images: {len(x_train)}, train labels: {len(y_train)}")
    print(f"valid images: {len(x_val)}, valid labels: {len(y_val)}")

    # transforms
    transform_train = transforms.Compose(
        [
            Gamma_2D(),
            Shift_2D(),
            RandomBrightness(),
            Rotation_2D(),
            RandomSharp(),
            RandomBlur(),
            RandomNoise(),
            Invert(),
            RandomClahe(),
            ToTensor(),
        ]
    )
    transform_test = transforms.Compose([ToTensor()])

    best_val_loss = [1.0] * NUM_OF_LANDMARKS

    for LN in range(NUM_OF_LANDMARKS):
        if str(LN) not in args.landmark_indices:
            continue

        print("Landmark index: ", LN)
        if args.wandb:
            import wandb

            wandb.init(
                project=args.wandb_project,
                config=args,
                name=f"{model_tag}_{LN}",
            )

        # dataloader - train
        dataset_train = Dental_Single_Data_Generator(
            (args.image_size, args.image_size),
            x_train,
            y_train,
            landmark_num=LN,
            mode="train",
            transform=transform_train,
        )
        dataloader_train = DataLoader(dataset_train, batch_size=args.train_batch_size, shuffle=True)

        # dataloader - test
        dataset_test = Dental_Single_Data_Generator(
            (args.image_size, args.image_size),
            x_val,
            y_val,
            landmark_num=LN,
            mode="train",
            transform=transform_test,
        )
        dataloader_test = DataLoader(dataset_test, batch_size=args.valid_batch_size, shuffle=False)

        # for j in range(0,1,1):
        #     sample = next(iter(dataloader_test))
        #     image = sample['image'][0]
        #     mask = sample['landmarks'][0]

        #     plt.figure(figsize=(10, 6))
        #     plt.subplot(1, 3, 1)
        #     image = np.squeeze(image,0)
        #     plt.imshow(image, 'gray')
        #     plt.subplot(1, 3, 2)
        #     mask = np.squeeze(mask,0)
        #     plt.imshow(mask,'gray')
        #     plt.subplot(1, 3, 3)
        #     print(image.shape, mask.shape)
        #     add = image + mask
        #     plt.imshow(add,'gray')
        #     plt.show()

        import segmentation_models_pytorch as smp

        model = smp.Unet(
            encoder_name=args.unet_encoder_name,
            decoder_attention_type=args.unet_decoder_attention_type,
            in_channels=args.unet_in_channels,
            classes=args.unet_classes,
        )
        #     try:
        #         weight_file = natsorted(glob.glob(HISTORY_PATH+str(LN)+'/*'))[-1]
        #         weight = torch.load(weight_file)
        #         model.load_state_dict(weight)
        #         print('load weight')
        #     except:
        #         print('no weight')

        model.to(device)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        best_acc_wts = copy.deepcopy(model.state_dict())

        optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
        mean_train_losses = []
        mean_val_losses = []

        for epoch in range(args.num_epochs):
            train_losses, val_losses = [], []
            train_dice, val_dice = [], []

            for batch_idx, sample in enumerate(Bar(dataloader_train)):

                images, masks = sample["image"].to(device), sample["landmarks"].to(device)

                outputs = model(images)
                outputs = torch.sigmoid(outputs)

                ddice_ = dice(outputs, masks)
                train_dice.append(ddice_.data.cpu().numpy())

                loss = dice_loss(outputs, masks)
                train_losses.append(loss.data.cpu().numpy())

                if args.wandb:
                    wandb.log(
                        {
                            "train/train_loss": loss.data.cpu().numpy(),
                            "train/train_dice": ddice_.data.cpu().numpy(),
                        }
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            print(
                "Epoch: {}. Train Dice:{:.5f}. Train Loss:{:.5f}".format(
                    epoch + 1,
                    np.mean(train_dice),
                    np.mean(train_losses),
                )
            )

            for batch_idx, sample in enumerate(Bar(dataloader_test)):
                images, masks = sample["image"].to(device), sample["landmarks"].to(device)

                outputs = model(images)
                outputs = torch.sigmoid(outputs)

                ddice_ = dice(outputs, masks)
                val_dice.append(ddice_.data.cpu().numpy())

                loss = dice_loss(outputs, masks)
                val_losses.append(loss.data.cpu().numpy())

            print(
                "Epoch: {}. Val Dice:{:.5f}. Val Loss:{:.5f}".format(
                    epoch + 1,
                    np.mean(val_dice),
                    np.mean(val_losses),
                )
            )

            mean_train_losses.append(np.mean(train_losses))
            mean_val_losses.append(np.mean(val_losses))

            # save model
            if np.mean(val_losses) < best_val_loss[LN]:
                best_val_loss[LN] = np.mean(val_losses)

                directory = os.path.join(model_save_path, str(LN))
                if not os.path.exists(directory):
                    os.makedirs(directory)

                filename = directory + "/" + "weight_best.pth"
                if torch.cuda.device_count() > 1:
                    torch.save(model.module.state_dict(), filename)
                else:
                    torch.save(model.module.state_dict(), filename)
                print("*" * 80)
                print("model_save!", filename)
                print("*" * 80)

            if args.wandb:
                wandb.log(
                    {
                        # "train/train_loss": np.mean(train_losses),
                        # "train_dice": np.mean(train_dice),
                        "valid/val_loss": np.mean(val_losses),
                        "valid/val_dice": np.mean(val_dice),
                        "valid/best_val_loss": best_val_loss[LN],
                        "epoch": epoch,
                    }
                )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple training script for training a Unet")

    # core
    parser.add_argument("--viewpos", help="viewpos", default="frontal", choices=["frontal", "lateral"])
    parser.add_argument("--landmark_indices", help="list of landmark indices", nargs="+", default=["0"])
    parser.add_argument("--image_size", help="image size", default=1024, choices=[384, 512, 1024], type=int)

    # directory
    parser.add_argument("--dataset_dir", help="Dataset directory path", required=True)
    parser.add_argument("--valid_dir", help="Dataset directory path", required=True)
    parser.add_argument("--model_save_path", help="Model save path")

    # model config
    parser.add_argument("--unet_encoder_name", help="unet encoder name", default="vgg16", choices=["vgg16", "timm-tf_efficientnet_lite4"])
    parser.add_argument("--unet_decoder_attention_type", help="unet decoder attention type", default=None, choices=[None, "scse"])
    parser.add_argument("--unet_in_channels", help="unet in channels", default=1, type=int)
    parser.add_argument("--unet_classes", help="unet classes", default=1, type=int)

    # train config
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--lr", help="Learning rate", default=1e-5, type=float)
    parser.add_argument("--num_epochs", help="Number of epochs", default=10, type=int)
    parser.add_argument("--train_batch_size", help="Batch size", default=4, type=int)
    parser.add_argument("--valid_batch_size", help="Batch size", default=1, type=int)

    # wandb config
    parser.add_argument("--wandb", help="Use wandb", action="store_true")
    parser.add_argument("--wandb_project", help="Wandb project name", default="flatnet_stage2")

    args = parser.parse_args()

    main(args)
