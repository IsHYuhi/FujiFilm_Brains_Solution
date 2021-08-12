import argparse
import os
import random
import time
from typing import Any, Dict, Tuple, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pytorch_widedeep.metrics import FBetaScore
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from libs.data_loader import SegImageDataset, SegImageTransform, seg_make_datapath_list
from libs.Loss import BinaryFocalLoss, DiceLoss
from libs.models import get_model
from libs.seg_config import Config, get_config

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
        prog="semantic segmentation",
        usage="python3 main.py",
        description="""
        This module demonstrates semantic segmentation using U-Net based Encoder-Decoder
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


def calc_acc_full(
    net: nn.Module, dataset: SegImageDataset, device: str
) -> Tuple[np.ndarray, np.ndarray]:
    net.eval()
    fbeta = FBetaScore(beta=0.5)
    accs = []
    fbs = []
    for i in range(len(dataset)):
        img, gt_mask = dataset[i]
        _, h, w = img.shape
        img = torch.unsqueeze(img, dim=0)
        gt_mask = gt_mask.to(device)

        with torch.no_grad():

            pred_mask = net(img.to(device))
            pred_mask = pred_mask.detach()

        b, c, h, w = pred_mask.shape
        sm = h * w
        fbeta.reset()
        pred_mask = torch.where(pred_mask > 0.5, 1, 0)
        accuracy = pred_mask.eq(gt_mask.detach()).sum().item() / sm
        fb = fbeta(pred_mask.view(-1, 1).float(), gt_mask.detach().view(-1, 1).float())

        accs.append(accuracy)
        fbs.append(fb)

    return np.mean(accs), np.mean(fbs)


def evaluate(
    net: nn.Module,
    dataset: SegImageDataset,
    device: str,
    filename: str,
    phase: str = "val",
    parser: Config = None,
) -> None:
    if phase == "val":
        img, gt_mask = zip(*[dataset[i] for i in range(20)])
        img = torch.stack(img)
        gt_mask = torch.stack(gt_mask)

        with torch.no_grad():
            pred_mask = net(img.to(device))
            pred_mask = pred_mask.to(device)

        # if you want to use threshold, when you use
        # pred_mask = torch.where(
        #     pred_mask.cpu().detach() >= torch.Tensor((0.5,)),
        #     torch.Tensor((255.0,)),
        #     torch.Tensor((0.0,)),
        # )

        grid_detect = make_grid(
            torch.cat((gt_mask.cpu().detach(), pred_mask.float().cpu().detach()), dim=0)
        )

    elif phase == "test":
        img = [dataset[i] for i in range(20)]
        img = torch.stack(img)

        with torch.no_grad():
            pred_mask = net(img.to(device))
            pred_mask = pred_mask.to(device)

        # if you want to use threshold, when you use
        # pred_mask = torch.where(
        #     pred_mask.cpu().detach() >= torch.Tensor((0.5,)),
        #     torch.Tensor((255.0,)),
        #     torch.Tensor((0.0,)),
        # )

        grid_detect = make_grid(pred_mask.float().cpu().detach())

    save_image(grid_detect, filename + "_mask.jpg")
    # for using images which have more than 3 channels
    img = img[:, 0:3, :, :]
    grid_img = make_grid(unnormalize(img))
    save_image(grid_img, filename + "_img.jpg")


def plot_log(data: Dict[str, Any], save_model_name: str = "model") -> None:
    plt.cla()
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.plot(data["net"], label="loss ")
    ax1.legend(loc="lower right")
    ax2 = ax1.twinx()
    ax2.plot(data["val_loss"], label="val_loss", color="orange")
    ax2.legend(loc="lower left")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax2.set_ylabel("val loss")
    ax1.set_title("train loss")
    plt.savefig("./logs/" + save_model_name + ".png")
    plt.close()


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


def check_dir(save_model_name: str) -> None:
    if not os.path.exists("./logs"):
        os.mkdir("./logs")
    if not os.path.exists("./logs/" + save_model_name[:-6]):
        os.mkdir("./logs/" + save_model_name[:-6])

    if not os.path.exists("./checkpoints"):
        os.mkdir("./checkpoints")
    if not os.path.exists("./checkpoints/" + save_model_name[:-6]):
        os.mkdir("./checkpoints/" + save_model_name[:-6])

    if not os.path.exists("./result"):
        os.mkdir("./result")
    if not os.path.exists("./result/" + save_model_name[:-6]):
        os.mkdir("./result/" + save_model_name[:-6])
    if not os.path.exists("./result/" + save_model_name):
        os.mkdir("./result/" + save_model_name)


def train_model(
    net: nn.Module,
    dataloaders: Dict[str, DataLoader],
    val_dataset: Dict[str, Any],
    test_dataset: Dict[str, Any],
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

    """binary"""
    criterion_cross_entropy = BinaryFocalLoss(gamma=2).to(device)
    criterion_cross_entropy2 = DiceLoss(beta=parser.dice_beta, with_bce=None).to(device)

    torch.backends.cudnn.benchmark = True

    net_losses = []
    val_losses = []
    # val_acc = []
    # val_fb = []
    # max_val_acc = 0
    # max_val_fb = 0.0
    min_val_loss = float("inf")
    stop_count = 0

    for epoch in range(parser.epoch + 1):
        if stop_count > 300:
            net.eval()
            return net
        stop_count += 1

        t_epoch_start = time.time()

        print("-----------")
        print("Epoch {}/{}".format(epoch, parser.epoch))
        # print("(train)")

        for phase in ["train", "val"]:
            if phase == "train":
                net.train()
            else:
                net.eval()

            batch_size = cast(int, dataloaders[phase].batch_size)
            epoch_net_loss = 0.0
            # epoch_FBloss = 0.0

            for images, gt_mask in tqdm(dataloaders[phase]):

                # if size of minibatch is 1, an error would be occured.
                # if you use Data parallel, an error might be occured.
                # You should use DDP and Sync Batch Norm. TODO
                if images.size()[0] == 1:
                    continue

                with torch.set_grad_enabled(phase == "train"):

                    images = images.to(device, non_blocking=True)
                    gt_mask = gt_mask.to(device, non_blocking=True)

                    mini_batch_size = images.size()[0]

                    pred_mask = net(images)
                    pred_mask = pred_mask.to(device)

                    # loss
                    net_loss = criterion_cross_entropy(
                        pred_mask, gt_mask.float()
                    )  # if you use focal loss or crossentropy(not BCE), attach .long().
                    net_loss += criterion_cross_entropy2(pred_mask, gt_mask.float())

                    if phase == "train":
                        net_loss.backward()
                        optimizer.step()

                    epoch_net_loss += net_loss.item() / mini_batch_size

            # epoch_val_acc, epoch_val_fb = calc_acc_full(net, val_dataset, device)

            # when using focal loss, some times loss explode.
            # TODO probablly when input image has only one class
            if phase == "train":
                net_losses += [epoch_net_loss / batch_size]
            if phase == "val":
                val_losses += [epoch_net_loss / batch_size]

        plot_log({"net": net_losses, "val_loss": val_losses}, save_model_name)

        if epoch % 500 == 0:
            torch.save(
                net.state_dict(),
                "checkpoints/" + save_model_name + "_" + str(epoch) + ".pth",
            )

        if min_val_loss > val_losses[-1]:
            stop_count = 0
            torch.save(
                net.state_dict(),
                "checkpoints/" + save_model_name + "_max_val_fb_net.pth",
            )
            min_val_loss = val_losses[-1]

        print("-----------")
        print(
            "epoch {} || Loss:{:.4f} || Val_loss:{:.4f} || Min_val_loss:{:.4f}".format(
                epoch, net_losses[-1], val_losses[-1], min_val_loss
            )
        )
        t_epoch_finish = time.time()
        print("timer: {:.4f} sec.".format(t_epoch_finish - t_epoch_start))

        t_epoch_start = time.time()

        net.eval()
        if epoch % 50 == 0:
            evaluate(
                net,
                val_dataset,
                device,
                "{:s}/val_{:d}".format("result/" + save_model_name, epoch),
                phase="val",
                parser=parser,
            )

            evaluate(
                net,
                test_dataset,
                device,
                "{:s}/test_{:d}".format("result/" + save_model_name, epoch),
                phase="test",
                parser=parser,
            )

    return net


def main(parser: Config, save_name: str) -> None:
    mean = (0.5,)
    std = (0.5,)

    for fold_id in range(0, 5):
        print("fold_{:d}".format(fold_id))

        net, weight_name = get_model(
            parser.model_name,
            parser.encoder_name,
            weight_name=None,
            in_channels=3,
            classes=parser.num_classes,
            activation="sigmoid",
        )

        train_img_list, val_img_list = seg_make_datapath_list(
            phase="train", n_splits=5, fold_id=fold_id, dataset_name=parser.dataset_name
        )
        test_img_list = seg_make_datapath_list(
            phase="test", dataset_name=parser.dataset_name
        )

        train_dataset = SegImageDataset(
            img_list=train_img_list,
            img_transform=SegImageTransform(size=parser.image_size, mean=mean, std=std),
            phase="train",
        )

        val_dataset = SegImageDataset(
            img_list=val_img_list,
            img_transform=SegImageTransform(size=parser.image_size, mean=mean, std=std),
            phase="val",
        )

        test_dataset = SegImageDataset(
            img_list=test_img_list,
            img_transform=SegImageTransform(size=parser.image_size, mean=mean, std=std),
            phase="test",
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=parser.batch_size,
            shuffle=True,
            num_workers=cast(int, os.cpu_count()),
        )

        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=parser.batch_size,
            shuffle=True,
            num_workers=cast(int, os.cpu_count()),
        )

        _ = train_model(
            net,
            dataloaders={"train": train_dataloader, "val": val_dataloader},
            val_dataset=val_dataset,
            test_dataset=test_dataset,
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
