import math
import os
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import albumentations as A
import cv2
import numpy as np
import pandas
import torch
import torch.utils.data as data
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import KFold, StratifiedKFold


def make_datapath_list(
    phase: str = "train",
    n_splits: int = 5,
    fold_id: Optional[int] = None,
    dataset_name: str = "Q2",
) -> Union[Tuple[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]:
    """
    make filepath list for train and validation image and label.
    """
    if fold_id is None:
        AssertionError("please make sure fold_id")

    if dataset_name == "Q3" and (phase == "train" or phase == "val"):
        rootpath = "./csv/{:s}_train_label.csv".format(dataset_name)
        path_list = pandas.read_csv(rootpath, header=None)

        path_train_list = {"img": list(path_list[0]), "label": list(path_list[1])}

        rootpath = "./csv/{:s}_test_label.csv".format(dataset_name[-2:])
        path_list = pandas.read_csv(rootpath, header=None)
        path_test_list = {"img": list(path_list[0]), "label": list(path_list[1])}

        return path_train_list, path_test_list

    elif dataset_name == "Q2" and phase == "train":
        rootpath = "./csv/{:s}_train_label.csv".format(dataset_name)
        path_list = pandas.read_csv(rootpath, header=None)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        for i, (train, val) in enumerate(
            cv.split(list(path_list[0]), list(path_list[1]))
        ):
            if fold_id == i:
                train_img = [list(path_list[0])[j] for j in train]
                val_img = [list(path_list[0])[j] for j in val]
                train_label = [list(path_list[1])[j] for j in train]
                val_label = [list(path_list[1])[j] for j in val]

        path_train_list = {"img": train_img, "label": train_label}
        path_val_list = {"img": val_img, "label": val_label}

        return path_train_list, path_val_list

    elif phase == "test":
        f = open("csv/Q2_test.txt", "r", encoding="UTF-8")
        path_list = [path[:-1] for path in f.readlines()]
        path_test_list = {"img": list(path_list[0])}
        f.close()

        return path_test_list

    return {}


def seg_make_datapath_list(
    phase: str = "train",
    n_splits: int = 5,
    fold_id: Optional[int] = None,
    dataset_name: str = "Q3",
) -> Union[Tuple[Dict[str, Any], Dict[str, Any]], Dict[str, Any]]:
    """
    make filepath list for train and validation image and mask.
    """
    if fold_id is None:
        AssertionError("please make sure fold_id")

    if dataset_name == "Q3":
        if phase == "train":

            path_list = os.listdir("./annotation/")
            path_list.sort()
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            for i, (train, val) in enumerate(cv.split(list(path_list))):
                if fold_id == i:
                    train_img = ["all_C_data/" + list(path_list)[j] for j in train]
                    val_img = ["all_C_data/" + list(path_list)[j] for j in val]
                    train_mask = ["annotation/" + list(path_list)[j] for j in train]
                    val_mask = ["annotation/" + list(path_list)[j] for j in val]

            path_train_list = {"img": train_img, "mask": train_mask}
            path_val_list = {"img": val_img, "mask": val_mask}

            return path_train_list, path_val_list

        elif phase == "test":
            rootpath = "./csv/Q3_test_label.csv"
            path_list = pandas.read_csv(rootpath, header=None)
            test_img = list(path_list[0])
            path_test_list = {"img": test_img}

            return path_test_list

    else:
        if phase == "train":
            rootpath = "./csv/Q2_seg_train.csv"
            path_list = pandas.read_csv(rootpath, header=None)
            cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            for i, (train, val) in enumerate(cv.split(list(path_list[0]))):
                if fold_id == i:
                    train_img = [list(path_list[0])[j] for j in train]
                    val_img = [list(path_list[0])[j] for j in val]
                    train_mask = [list(path_list[1])[j] for j in train]
                    val_mask = [list(path_list[1])[j] for j in val]

            path_train_list = {"img": train_img, "mask": train_mask}
            path_val_list = {"img": val_img, "mask": val_mask}

            return path_train_list, path_val_list

        elif phase == "test":
            f = open("csv/Q2_test.txt", "r", encoding="UTF-8")
            path_list = [path[:-1] for path in f.readlines()]
            path_test_list = {"img": list(path_list)}
            f.close()

            return path_test_list

    return {}


class ImageTransform:
    def __init__(self, size: int, mean: Tuple, std: Tuple) -> None:
        self.data_transform = {
            "train": A.Compose(
                [
                    A.Resize(size, size),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Transpose(p=0.5),
                    A.CoarseDropout(p=0.5),
                    A.ColorJitter(p=0.5),
                    A.GaussNoise(p=0.5),
                    A.GaussianBlur(p=0.5),
                    A.Normalize(mean, std),
                    ToTensorV2(),
                ]
            ),
            "val": A.Compose(
                [A.Resize(size, size), A.Normalize(mean, std), ToTensorV2()]
            ),
            "test": A.Compose(
                [A.Resize(size, size), A.Normalize(mean, std), ToTensorV2()]
            ),
        }

    def __call__(
        self, image: Union[List, np.ndarray], phase: str
    ) -> Dict[str, torch.Tensor]:
        if isinstance(image, list):
            return self.data_transform[phase](image=image[0], mask=image[1])
        else:
            return self.data_transform[phase](image=image)


class SegImageTransform:
    def __init__(self, size: int, mean: Tuple, std: Tuple) -> None:
        self.data_transform = {
            "train": A.Compose(
                [
                    A.Resize(size, size),
                    A.Rotate(interpolation=cv2.INTER_NEAREST, p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.Transpose(p=0.5),
                    A.CoarseDropout(p=0.5),
                    A.RandomGridShuffle(p=0.5),  # Q2:off Q3:on
                    A.RandomResizedCrop(size, size, interpolation=cv2.INTER_NEAREST),
                    # A.ElasticTransform(p=0.5),
                    # A.GridDistortion(p=0.5),
                    # A.Perspective(p=0.5),
                    A.ColorJitter(p=0.5),
                    A.GaussNoise(p=0.5),
                    A.GaussianBlur(p=0.5),
                    A.Normalize(mean, std),
                    ToTensorV2(),
                ]
            ),
            "val": A.Compose(
                [A.Resize(size, size), A.Normalize(mean, std), ToTensorV2()]
            ),
            "test": A.Compose(
                [A.Resize(size, size), A.Normalize(mean, std), ToTensorV2()]
            ),
        }

    def __call__(
        self, image: Union[List, np.ndarray], phase: str
    ) -> Dict[str, torch.Tensor]:
        if isinstance(image, list):
            return self.data_transform[phase](image=image[0], mask=image[1])
        else:
            return self.data_transform[phase](image=image)


class ImageDataset(data.Dataset):
    """
    Dataset class. Inherit Dataset class from PyTrorch.
    """

    def __init__(
        self,
        img_list: Dict[str, Any],
        img_transform: ImageTransform,
        phase: str = "train",
    ) -> None:
        self.img_list = img_list
        self.img_transform = img_transform
        self.phase = phase

    def __len__(self) -> int:
        return len(self.img_list["img"])

    def __getitem__(
        self, index: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, int], None]:
        """
        get tensor type preprocessed Image
        """
        img = cv2.imread(self.img_list["img"][index])

        if self.phase == "train" or self.phase == "val":
            label = self.img_list["label"][index]
            img = self.img_transform(img, self.phase)
            # print(torch.max(img), torch.min(img))
            return img["image"], label

        elif self.phase == "test":
            label = self.img_list["label"][index]
            img = self.img_transform(img, self.phase)
            return img["image"], label

        return None


class SegImageDataset(data.Dataset):
    """
    Dataset class. Inherit Dataset class from PyTrorch.
    """

    def __init__(
        self,
        img_list: Dict[str, Any],
        img_transform: SegImageTransform,
        phase: str = "train",
    ) -> None:
        self.img_list = img_list
        self.img_transform = img_transform
        self.phase = phase

    def __len__(self) -> int:
        return len(self.img_list["img"])

    def __getitem__(
        self, index
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], None]:
        """
        get tensor type preprocessed Image
        """
        img = cv2.imread(self.img_list["img"][index])

        if self.phase == "train" or self.phase == "val":
            mask = cv2.imread(self.img_list["mask"][index], 0) // 255
            while True:
                res = self.img_transform([img, mask], self.phase)
                torch.max(res["mask"]) == 1
                break

            return res["image"], torch.unsqueeze(res["mask"], dim=0)

        elif self.phase == "test":
            img = self.img_transform(img, self.phase)
            return img["image"]

        return None


class BinaryBalancedSampler:
    def __init__(
        self, img_list: Dict[str, Any], img_transform: ImageTransform, n_samples: int
    ) -> None:
        self.img_transform = img_transform
        self.features, self.labels = np.array(img_list["img"]), np.array(
            img_list["label"]
        )

        label_counts = np.bincount(self.labels)

        major_label = label_counts.argmax()
        minor_label = label_counts.argmin()

        self.major_indices = np.where(self.labels == major_label)[0]
        self.minor_indices = np.where(self.labels == minor_label)[0]

        np.random.shuffle(self.major_indices)
        np.random.shuffle(self.minor_indices)

        self.used_indices = 0
        self.count = 0
        self.n_samples = n_samples
        self.batch_size = self.n_samples * 2

    def __iter__(self) -> Generator:
        self.count = 0
        while self.used_indices + self.n_samples < len(self.minor_indices):
            # 多数派データ(major_indices)からはランダムに選び出す操作を繰り返す
            # 少数派データ(minor_indices)からは順番に選び出し
            indices = (
                self.minor_indices[
                    self.used_indices : self.used_indices + self.n_samples
                ].tolist()
                + np.random.choice(
                    self.major_indices, self.n_samples, replace=False
                ).tolist()
            )
            yield self.get_img(self.features[indices]), torch.tensor(
                self.labels[indices]
            )

            self.used_indices += self.n_samples
            self.count += self.n_samples

        indices = self.minor_indices[self.used_indices :].tolist()
        indices += np.random.choice(
            self.major_indices, len(indices), replace=False
        ).tolist()
        yield self.get_img(self.features[indices]), torch.tensor(self.labels[indices])

        self.count = 0
        self.used_indices = 0

    def __len__(self) -> int:
        return int(math.ceil(len(self.minor_indices) / (self.n_samples)))

    def get_img(self, paths: List[str]) -> torch.Tensor:
        b_img = []
        for path in paths:
            img = cv2.imread(path)
            res = self.img_transform(img, "train")
            img = res["image"]
            img = torch.Tensor(img)
            b_img.append(img)

        return torch.stack(b_img)
