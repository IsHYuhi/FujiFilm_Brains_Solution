import argparse
import glob
import os
from typing import Any, Dict

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch import Tensor
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from libs.data_loader import seg_make_datapath_list
from libs.grid_crop import GridCrop
from libs.models import fix_model_state_dict, get_model
from libs.seg_config import Config, get_config
from train import seed_everything

seed_everything(seed=42)


def get_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="semantic classification inference",
        usage="python3 Q2_inference.py",
        description="""
        This module demonstrates classification inference.
        """,
        add_help=True,
    )

    parser.add_argument("config", type=str, help="path of a config file")
    parser.add_argument("--wall-type", type=str, default="W,D,P", help="wall type")
    parser.add_argument(
        "--threshold", type=float, default="0.5", help="threshold for baggin"
    )
    parser.add_argument("--save-full-image", action="store_true")
    parser.add_argument("--sub-pcon", type=str, default=None)

    return parser.parse_args()


def get_tta_dic(h: int, w: int) -> Dict[str, Dict[str, Any]]:
    tta_dic = {
        "normal": {
            "transform": A.Compose(
                [
                    A.Normalize(0.5, 0.5),
                    ToTensorV2(),
                ]
            ),
            "untransform": transforms.Resize(
                (h, w), interpolation=InterpolationMode.NEAREST
            ),
        },
        "hflip": {
            "transform": A.Compose(
                [
                    A.HorizontalFlip(p=1),
                    A.Normalize(0.5, 0.5),
                    ToTensorV2(),
                ]
            ),
            "untransform": transforms.Compose([transforms.RandomHorizontalFlip(p=1.0)]),
        },
        "vflip": {
            "transform": A.Compose(
                [
                    A.VerticalFlip(p=1),
                    A.Normalize(0.5, 0.5),
                    ToTensorV2(),
                ]
            ),
            "untransform": transforms.Compose([transforms.RandomVerticalFlip(p=1.0)]),
        },
        "vhflip": {
            "transform": A.Compose(
                [
                    A.VerticalFlip(p=1),
                    A.HorizontalFlip(p=1),
                    A.Normalize(0.5, 0.5),
                    ToTensorV2(),
                ]
            ),
            "untransform": transforms.Compose(
                [
                    transforms.RandomVerticalFlip(p=1.0),
                    transforms.RandomHorizontalFlip(p=1.0),
                ]
            ),
        },
        "halfscale": {
            "transform": A.Compose(
                [
                    A.Resize(h // 2, w // 2, interpolation=cv2.INTER_NEAREST),
                    A.Normalize(0.5, 0.5),
                    ToTensorV2(),
                ]
            ),
            "untransform": transforms.Resize(
                (h, w), interpolation=InterpolationMode.NEAREST
            ),
        },
        "x2scale": {
            "transform": A.Compose(
                [
                    A.Resize(h * 2, w * 2, interpolation=cv2.INTER_NEAREST),
                    A.Normalize(0.5, 0.5),
                    ToTensorV2(),
                ]
            ),
            "untransform": transforms.Resize(
                (h, w), interpolation=InterpolationMode.NEAREST
            ),
        },
    }
    return tta_dic


def sliding_window(
    net: nn.Module, device: str, num: str, size: int, stride: int, root_path: str
) -> Tensor:

    img = cv2.imread("{:s}/reconst/{:s}.png".format(root_path, num))
    h, w, _ = img.shape
    tta_dic = get_tta_dic(h, w)

    transformed = tta_dic["normal"]["transform"](image=img)["image"]
    # print("patched image size: ", transformed.shape)
    grid_crop = GridCrop(img=transformed, grid_size=(size, size), stride=stride)
    grid_img = grid_crop.forward()
    b, _, _, _ = grid_img.shape
    # print("grid image shape: ", grid_img.shape)

    mask_grid = []
    for i in range(b):
        pred = net(grid_img[i : i + 1, :, :, :].to(device))
        mask_grid.append(pred.squeeze(dim=0).detach().cpu())

    tta = torch.stack(mask_grid)
    tta = grid_crop.sliding_window(tta.detach().clone().cpu())
    # tta = tta_dic['normal']['untransform'](tta)
    mask = tta

    ttas = ["hflip", "vflip"]
    for t in ttas:
        # print(t)
        transformed = tta_dic[t]["transform"](image=img)["image"]
        grid_crop = GridCrop(img=transformed, grid_size=(size, size), stride=stride)
        grid_img = grid_crop.forward()
        b, _, _, _ = grid_img.shape
        mask_grid = []
        for i in range(b):
            pred = net(grid_img[i : i + 1, :, :, :].to(device))
            mask_grid.append(pred.squeeze(dim=0).detach().cpu())

        tta = torch.stack(mask_grid)
        tta = grid_crop.sliding_window(tta.detach().clone().cpu())

        tta = tta_dic[t]["untransform"](tta)
        mask = mask + tta

    mask /= 1 + len(ttas)
    return mask


def make_submission(sub_dir: str, load_dir: str) -> None:
    if not os.path.exists(sub_dir):
        os.mkdir(sub_dir)
    test_img_list = seg_make_datapath_list(phase="test")
    i = 0
    for name in test_img_list["img"]:
        if "_" in name:
            new_name = name.split("_")[0] + ".jpg"
        else:
            new_name = name

        img = (
            Image.fromarray(
                np.array(
                    Image.open(
                        load_dir + new_name.split("/")[-1].split(".")[0] + ".png"
                    ),
                    dtype=np.uint8,
                )
                // 255
            )
            if test_img_list["img"][i].split("/")[1][0] == "C"
            else Image.fromarray(
                np.zeros_like(
                    np.array(
                        Image.open(
                            load_dir + new_name.split("/")[-1].split(".")[0] + ".png"
                        ),
                        dtype=np.uint8,
                    )
                )
            )
        )
        img.save(
            "{:s}/".format(sub_dir) + name.split("/")[-1].split(".")[0] + ".png",
            mode="L",
        )
        i += 1


def main(
    config: Config, parser: argparse.Namespace, load_model_name: str, wall_type: str
) -> None:
    numbers = glob.glob("./{:s}/reconst/*".format(wall_type))
    numbers = sorted([num.split("/")[-1].split(".")[0] for num in numbers])
    post_process_list = glob.glob(
        "./{:s}/C{:s}/*".format(parser.wall_type, parser.wall_type)
    )
    post_process_list = [
        path.split("/")[-1].split(".")[0].split("-")[0]
        + "-"
        + path.split("/")[-1].split(".")[0].split("-")[1]
        for path in post_process_list
    ]
    print("the number of {:s} type is {:d}".format(parser.wall_type, len(numbers)))
    width = 13 if wall_type == "P" else 14

    dir = "submission_segmentation/{:s}".format(load_model_name)
    if not os.path.exists(dir + "_full") and (
        config.dataset_name == "Q2" or parser.save_full_image
    ):
        os.mkdir(dir + "_full")
    if not os.path.exists(dir) and config.dataset_name == "Q3":
        os.mkdir(dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    for num in numbers:
        print(num)
        pred_masks_fold = []
        for fold_id in tqdm(range(5)):
            # print("fold {:d}".format(fold_id))

            net, _ = get_model(
                config.model_name,
                config.encoder_name,
                weight_name=None,
                in_channels=3,
                classes=config.num_classes,
                activation="sigmoid",
            )
            checkpoint_model_name = "{:s}/fold{:d}".format(
                load_model_name,
                fold_id,
            )
            net_weights = torch.load(
                "./checkpoints/" + checkpoint_model_name + "_max_val_fb_net.pth",
                map_location=torch.device(device),
            )
            net.load_state_dict(fix_model_state_dict(net_weights))
            net.to(device)
            net.eval()
            pred_mask = sliding_window(
                net, device, num, config.image_size, config.image_size // 2, wall_type
            )
            pred_masks_fold.append(pred_mask)

        mask: Tensor = pred_masks_fold[0]
        for fold_id in range(1, 5):
            mask = mask + pred_masks_fold[fold_id]

        avg_mask: Tensor = mask / 5

        save_fig = np.where(
            avg_mask.squeeze().detach().cpu().numpy() > parser.threshold, 255, 0
        )

        if config.dataset_name == "Q2" or parser.save_full_image:
            cv2.imwrite("{:s}_full/{:s}.png".format(dir, str(num)), save_fig)

        if (parser.sub_pcon is not None) and (wall_type == "W"):
            pcon = cv2.imread(
                "./submission_segmentation/{:s}_full/{:s}.png".format(
                    parser.sub_pcon, str(num)
                ),
                0,
            )
            save_fig = save_fig - pcon
            save_fig = np.where(save_fig < 0, 0, save_fig)

        if config.dataset_name == "Q3":
            count = 1
            for h in range(width):
                for w in range(18):
                    if (
                        "{:s}-{:s}".format(str(num).zfill(3), str(count))
                        not in post_process_list
                    ):
                        save_fig[h * 256 : (h + 1) * 256, w * 256 : (w + 1) * 256] = 0
                    cv2.imwrite(
                        "{:s}/{:s}-{:s}.png".format(dir, str(num).zfill(3), str(count)),
                        save_fig[h * 256 : (h + 1) * 256, w * 256 : (w + 1) * 256],
                    )
                    count += 1

            if parser.save_full_image:
                cv2.imwrite("{:s}_full/{:s}.png".format(dir, str(num)), save_fig)


if __name__ == "__main__":
    parser = get_parser()
    load_model_name = parser.config.split("/")[-1].split(".")[0]
    config = get_config(parser.config)
    for wall_type in parser.wall_type.split(","):
        main(config, parser, load_model_name, wall_type)

    if config.dataset_name == "Q3":
        root_dir = "./submission_segmentation"
        make_submission(
            "{:s}/submission_{:s}/".format(root_dir, load_model_name),
            "{:s}/{:s}/".format(root_dir, load_model_name),
        )
