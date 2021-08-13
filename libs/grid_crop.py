import math
from typing import Generator, Optional, Tuple

import torch
from torch import Tensor


class GridCrop:
    def __init__(
        self, img: Tensor, grid_size: Tuple[int, int] = (256, 256), stride: int = 128
    ) -> None:
        super().__init__()
        self.img = img
        _, self.im_h, self.im_w = img.shape
        self.stride = stride
        self.height = grid_size[0]
        self.width = grid_size[1]
        self.ones = torch.ones((self.height, self.width), dtype=img.dtype)
        self.zeros = torch.zeros((1, self.im_h, self.im_w), dtype=img.dtype)
        self.out = torch.zeros((1, self.im_h, self.im_w), dtype=img.dtype)
        self.h_c = 0
        self.w_c = 0

    def get_grid(self) -> Generator:
        h_c = math.ceil(self.im_h / self.stride)
        w_c = math.ceil(self.im_w / self.stride)
        h_flag = False
        for h1 in range(h_c):
            self.h_c += 1
            self.w_c = 0
            w_flag = False
            for w1 in range(w_c):
                self.w_c += 1
                h2 = h1 * self.stride
                w2 = w1 * self.stride
                if h2 + self.height > self.im_h:
                    h2 = self.im_h - self.height
                    h_flag = True
                if w2 + self.width > self.im_w:
                    w2 = self.im_w - self.width
                    w_flag = True
                yield self.crop(h2, w2)
                if w_flag:
                    break
            if h_flag and w_flag:
                break

    def crop(self, top: int, left: int) -> Tensor:
        return self.img[..., top : top + self.height, left : left + self.width]

    def make_grid(self, img: Tensor) -> Tensor:
        h_c = self.im_h // self.height + 1
        w_c = self.im_w // self.width + 1
        for h1 in range(h_c):
            row_image = img[h1 * w_c : h1 * w_c + 1, :, :, :]
            for w1 in range(w_c):
                if w1 == 0:
                    continue
                elif w1 == w_c - 1:
                    row_image = torch.cat(
                        [
                            row_image,
                            img[
                                h1 * w_c + w1 : h1 * w_c + w1 + 1,
                                :,
                                :,
                                (1 + w1) * self.width - self.im_w :,
                            ],
                        ],
                        dim=3,
                    )
                else:
                    row_image = torch.cat(
                        [row_image, img[h1 * w_c + w1 : h1 * w_c + w1 + 1, :, :, :]],
                        dim=3,
                    )
            if h1 == 0:
                all_image = row_image
            elif h1 == h_c - 1:
                all_image = torch.cat(
                    [
                        all_image,
                        row_image[:, :, (1 + h1) * self.height - self.im_h :, :],
                    ],
                    dim=2,
                )
            else:
                all_image = torch.cat([all_image, row_image], dim=2)
        return all_image

    def sliding_window(self, img: Tensor, th: Optional[int] = None):
        for h1 in range(self.h_c):
            for w1 in range(self.w_c):
                h2 = h1 * self.stride
                w2 = w1 * self.stride
                if h2 + self.height > self.im_h:
                    h2 = self.im_h - self.height
                if w2 + self.width > self.im_w:
                    w2 = self.im_w - self.width
                self.out[:, h2 : h2 + self.height, w2 : w2 + self.width] += img[
                    h1 * self.w_c + w1, :, :, :
                ]
                self.zeros[:, h2 : h2 + self.height, w2 : w2 + self.width] += self.ones
        self.out = self.out / self.zeros

        if th:
            self.out = torch.where(
                self.out >= torch.Tensor((th,)),
                torch.Tensor((255.0,)),
                torch.Tensor((0.0,)),
            )
        return self.out

    def forward(self) -> Tensor:
        img = [img for i, img in enumerate(self.get_grid())]
        stack_image = torch.stack(img)
        return stack_image
