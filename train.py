# CUDA_VISIBLE_DEVICES=0 python3 train.py ...
import argparse
import os
import random
import time
from typing import Any, Dict, List, Tuple, Union, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from libs.config import Config, get_config
from libs.data_loader import (
    BinaryBalancedSampler,
    ImageDataset,
    ImageTransform,
    make_datapath_list,
)
from libs.models import get_fcn

matplotlib.use("Agg")


def seed_everything(seed=42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


seed_everything(seed=42)


def get_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="semantic classification",
        usage="python3 train.py",
        description="""
        This module demonstrates classification.
        """,
        add_help=True,
    )

    parser.add_argument("config", type=str, help="path of a config file")

    return parser.parse_args()


def set_requires_grad(nets: nn.Module, requires_grad: bool = False) -> None:
    for net in [nets]:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def unnormalize(x: torch.Tensor) -> torch.Tensor:
    x = x.transpose(1, 3)
    # mean, std
    x = x * torch.Tensor((0.5,)) + torch.Tensor((0.5,))
    x = x.transpose(1, 3)
    return x


def calc_acc_f1(
    net: nn.Module, dataset: ImageDataset, device: str
) -> Tuple[np.ndarray, np.ndarray]:
    net.eval()
    preds = []
    labels = []
    for i in range(dataset.__len__()):
        img, label = dataset[i]
        _, h, w = img.shape
        img = torch.unsqueeze(img, dim=0)

        with torch.no_grad():
            pred = net(img.to(device))
            _, pred = torch.max(pred, dim=1)
            preds.append(pred.squeeze().detach().cpu().numpy())
            labels.append(label)

    preds = np.array(preds)
    labels = np.array(labels)

    return accuracy_score(labels, preds), f1_score(labels, preds)


def plot_log(data, save_model_name="model"):
    plt.cla()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(data["net"], label="loss ")
    ax1.legend(loc="lower right")
    ax2 = ax1.twinx()
    ax2.plot(data["val_acc"], label="val_acc", color="green")
    ax2.plot(data["val_f1"], label="val_f1", color="orange")
    ax2.legend(loc="lower left")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax2.set_ylabel("Accuracy&FB")
    ax1.set_title("Loss&acc&FB")
    plt.savefig("./logs/" + save_model_name + ".png")
    plt.close()


def check_dir(save_model_name: str) -> None:
    if not os.path.exists("./logs"):
        os.mkdir("./logs")
    if not os.path.exists("./logs/" + save_model_name[:-6]):
        os.mkdir("./logs/" + save_model_name[:-6])

    if not os.path.exists("./checkpoints"):
        os.mkdir("./checkpoints")
    if not os.path.exists("./checkpoints/" + save_model_name[:-6]):
        os.mkdir("./checkpoints/" + save_model_name[:-6])


def get_optimizer(net: nn.Module, parser: Config) -> Tuple[nn.Module, Any]:
    optimizer: torch.optim.Optimizer
    if parser.optimizer == "Adam":
        beta1, beta2 = 0.5, 0.999
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=parser.learning_rate,
            betas=(beta1, beta2),
            weight_decay=1e-5,
        )
    elif parser.optimizer == "Adadelta":
        optimizer = torch.optim.Adadelta(
            filter(lambda p: p.requires_grad, net.parameters()),
            weight_decay=1e-5,
        )
    elif parser.optimizer == "SGD":
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, net.parameters()),
            lr=parser.learning_rate,
            momentum=0.9,
            weight_decay=1e-5,
        )

    return net, optimizer


def train_model(
    net: nn.Module,
    dataloaders: Dict[str, Union[DataLoader, BinaryBalancedSampler]],
    parser: Config,
    save_model_name: str = "model",
) -> nn.Module:

    check_dir(save_model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    net.to(device)

    """use GPU in parallel"""
    if device == "cuda":
        net = torch.nn.DataParallel(net)
        print("parallel mode")

    print("device:{}".format(device))

    net, optimizer = get_optimizer(net, parser)

    criterion_cross_entropy = nn.CrossEntropyLoss()

    torch.backends.cudnn.benchmark = True

    net_losses: List[float] = []
    val_acc: List[float] = []
    val_f1: List[float] = []
    max_val_acc: float = 0.0
    max_val_f1: float = 0.0
    stop_count: int = 0

    for epoch in range(parser.epoch + 1):
        if stop_count > 500:
            net.eval()
            return net
        stop_count += 1

        t_epoch_start = time.time()

        print("-----------")
        print("Epoch {}/{}".format(epoch, parser.epoch))

        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()

            if phase == "val" and epoch <= -1:
                stop_count = 0
                val_acc += [0.0]
                val_f1 += [0.0]
                continue

            batch_size: int = cast(int, dataloaders[phase].batch_size)

            epoch_net_loss: float = 0.0
            epoch_acc: float = 0.0
            epoch_f1: float = 0.0
            epoch_preds = []
            epoch_labels = []

            for images, labels in tqdm(dataloaders[phase]):

                optimizer.zero_grad()

                # if size of minibatch is 1, an error would be occured.
                # if you use Data parallel, an error might be occured.
                # You should use DDP and Sync Batch Norm. TODO
                if images.size()[0] == 1:
                    continue

                with torch.set_grad_enabled(phase == "train"):

                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    mini_batch_size = images.size()[0]

                    preds = net(images)

                    # loss
                    net_loss = criterion_cross_entropy(preds, labels)
                    _, preds = torch.max(preds, 1)  # ラベルを予測

                    if phase == "train":
                        net_loss.backward()
                        optimizer.step()

                    epoch_net_loss += net_loss.item() / mini_batch_size

                epoch_preds.extend(preds.detach().cpu().numpy())
                epoch_labels.extend(labels.detach().cpu().numpy())

            epoch_preds = np.array(epoch_preds)
            epoch_labels = np.array(epoch_labels)

            epoch_acc = accuracy_score(epoch_labels, epoch_preds)
            epoch_f1 = f1_score(epoch_labels, epoch_preds)

            # epoch_val_acc, epoch_val_f1 = calc_acc_f1(net, val_dataset, device)

            # when using focal loss, some times g_loss explode. TODO
            # probablly when input image has only one class?
            if phase == "train":
                net_losses += [epoch_net_loss / batch_size]
            if phase == "val":
                val_acc += [epoch_acc]
                val_f1 += [epoch_f1]

        plot_log(
            {"net": net_losses, "val_acc": val_acc, "val_f1": val_f1}, save_model_name
        )

        if epoch % 500 == 0:
            torch.save(
                net.state_dict(),
                "checkpoints/" + save_model_name + "_" + str(epoch) + ".pth",
            )

        if max_val_acc < val_acc[-1]:
            torch.save(
                net.state_dict(),
                "checkpoints/" + save_model_name + "_max_val_acc_net.pth",
            )
            max_val_acc = val_acc[-1]

        if max_val_f1 < val_f1[-1]:
            stop_count = 0
            torch.save(
                net.state_dict(),
                "checkpoints/" + save_model_name + "_max_val_f1_net.pth",
            )
            max_val_f1 = val_f1[-1]

        print("-----------")
        print(
            "epoch {} || Loss:{:.4f} || Val_acc:{:.4f} || ".format(
                epoch, net_losses[-1], val_acc[-1]
            )
            + "Val_f1:{:.4f} || Max_val_f1:{:.4f}".format(val_f1[-1], max_val_f1)
        )

        t_epoch_finish = time.time()
        print("timer: {:.4f} sec.".format(t_epoch_finish - t_epoch_start))

    return net


def main(parser, save_name):
    mean = (0.5,)
    std = (0.5,)

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

        train_img_list, val_img_list = make_datapath_list(
            phase="train", n_splits=5, fold_id=fold_id, dataset_name=parser.dataset_name
        )

        # train_dataset = ImageDataset(
        #     img_list=train_img_list,
        #     img_transform=ImageTransform(size=parser.image_size, mean=mean, std=std),
        #     phase="train",
        # )

        val_dataset = ImageDataset(
            img_list=val_img_list,
            img_transform=ImageTransform(size=parser.image_size, mean=mean, std=std),
            phase="val",
        )

        # train_dataloader = DataLoader(
        #     train_dataset,
        #     batch_size=parser.batch_size,
        #     shuffle=True,
        #     num_workers=cast(int, os.cpu_count()),
        # )

        train_dataloader = BinaryBalancedSampler(
            img_list=train_img_list,
            img_transform=ImageTransform(size=parser.image_size, mean=mean, std=std),
            n_samples=parser.batch_size // 2,
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=parser.batch_size,
            shuffle=True,
            num_workers=cast(int, os.cpu_count()),
        )

        dataloaders = {"train": train_dataloader, "val": val_dataloader}

        _ = train_model(
            net,
            dataloaders=dataloaders,
            parser=parser,
            save_model_name="{:s}/fold{:d}".format(
                save_name,
                fold_id,
            ),
        )


if __name__ == "__main__":
    parser = get_parser()
    save_name = parser.config.split("/")[-1].split(".")[0]
    config = get_config(parser.config)
    main(config, save_name)
