import argparse
import os
from typing import List, Union

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn

from libs.config import Config, get_config
from libs.data_loader import ImageDataset, ImageTransform, make_datapath_list
from libs.models import fix_model_state_dict, get_fcn
from train import calc_acc_f1, seed_everything

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
    parser.add_argument(
        "--top-k", type=int, default=5, help="top-k of validation score"
    )

    return parser.parse_args()


def calc_ensemble(preds: np.ndarray, th: int) -> np.ndarray:
    ens_preds = np.zeros_like(preds[0])
    for pred in preds:
        ens_preds += pred

    ens_preds = np.where(ens_preds >= th, 1, 0)

    return ens_preds


def predict(net: nn.Module, dataset: ImageDataset, device: str) -> np.ndarray:
    net.eval()
    preds = []
    for i in range(dataset.__len__()):
        img = dataset[i]
        _, h, w = img.shape
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            pred = net(img.to(device))
            _, pred = torch.max(pred, dim=1)
            preds.append(pred.squeeze().detach().cpu().numpy())

    preds = np.array(preds)

    return preds


def save_csv(preds: Union[List, np.ndarray], img_list: List, save_name: str) -> None:
    submit = pd.DataFrame()
    submit["img"] = img_list
    submit["label"] = list(preds)
    submit.to_csv("submission_csv/{:s}.csv".format(save_name), header=None, index=None)


def main(parser: Config, load_model_name: str, top_k: int) -> None:
    if not os.path.exists("./submission_csv"):
        os.mkdir("./submission_csv")
    ens = []
    preds = []
    mean = (0.5,)
    std = (0.5,)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_img_list = make_datapath_list(phase="test", dataset_name=parser.dataset_name)
    test_dataset = ImageDataset(
        img_list=test_img_list,
        img_transform=ImageTransform(size=parser.image_size, mean=mean, std=std),
        phase="test",
    )

    for fold_id in range(0, 5):
        print("fold_{:d}".format(fold_id))

        if parser.model_name == "fcn":
            net = get_fcn(
                pretrained=False, in_channels=3, out_channels=parser.num_classes
            )
        else:
            net = timm.create_model(
                parser.model_name, pretrained=False, num_classes=parser.num_classes
            )

        _, val_img_list = make_datapath_list(
            phase="train", n_splits=5, fold_id=fold_id, dataset_name=parser.dataset_name
        )

        val_dataset = ImageDataset(
            img_list=val_img_list,
            img_transform=ImageTransform(size=parser.image_size, mean=mean, std=std),
            phase="val",
        )

        checkpoint_model_name = "{:s}/fold{:d}".format(
            load_model_name,
            fold_id,
        )

        save_csv_name = "{:s}_fold{:d}".format(load_model_name, fold_id)

        net_weights = torch.load(
            "./checkpoints/" + checkpoint_model_name + "_max_val_f1_net.pth",
            map_location=torch.device(device),
        )
        net.load_state_dict(fix_model_state_dict(net_weights))
        net.to(device)
        net.eval()
        val_acc, val_f1 = calc_acc_f1(net, val_dataset, device)
        print(
            "validation accuracy: {:f} || validation f1 score: {:f}".format(
                val_acc, val_f1
            )
        )
        test_preds = predict(net, test_dataset, device)
        save_csv(test_preds, test_img_list["img"], save_csv_name)
        preds.append([test_preds, val_f1])

    preds = sorted(preds, key=lambda x: x[1], reverse=True)
    ens = calc_ensemble([x[0] for x in preds[:top_k]], top_k)
    save_csv(ens, test_img_list["img"], load_model_name)


if __name__ == "__main__":
    parser = get_parser()
    save_name = parser.config.split("/")[-1].split(".")[0]
    config = get_config(parser.config)
    main(config, save_name, parser.top_k)
