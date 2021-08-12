from typing import Any, List, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported
    which is proposed in 'Focal Loss for Dense Object Detection.
    (https://arxiv.org/abs/1708.02002)' Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified
    examples (p>0.5) putting more focus on hard misclassified example

    :param reduction: `none`|`mean`|`sum`
    :param **kwargs
        balance_index: (int) balance class index, should be specific when alpha is float
    """

    def __init__(
        self,
        alpha: Union[List[float], np.ndarray] = [1.0, 1.0],
        gamma: int = 2,
        ignore_index: Any = None,
        reduction: str = "mean",
    ) -> None:
        super(BinaryFocalLoss, self).__init__()
        if alpha is None:
            alpha = [0.25, 0.75]
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = 1e-6
        self.ignore_index = ignore_index
        self.reduction = reduction

        assert self.reduction in ["none", "mean", "sum"]

        if self.alpha is None:
            self.alpha = torch.ones(2)
        elif isinstance(self.alpha, (list, np.ndarray)):
            self.alpha = np.asarray(self.alpha)
            self.alpha = np.reshape(self.alpha, (2))
            assert (
                self.alpha.shape[0] == 2
            ), "the `alpha` shape is not match the number of class"
        elif isinstance(self.alpha, (float, int)):
            self.alpha = np.asarray(
                [self.alpha, 1.0 - self.alpha], dtype=np.float
            ).view(2)

        else:
            raise TypeError("{} not supported".format(type(self.alpha)))

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = torch.sigmoid(output)
        prob = torch.clamp(prob, self.smooth, 1.0 - self.smooth)

        pos_mask = (target == 1).float()
        neg_mask = (target == 0).float()

        pos_loss = (
            -self.alpha[0]
            * torch.pow(torch.sub(1.0, prob), self.gamma)
            * torch.log(prob)
            * pos_mask
        )
        neg_loss = (
            -self.alpha[1]
            * torch.pow(prob, self.gamma)
            * torch.log(torch.sub(1.0, prob))
            * neg_mask
        )

        neg_loss = neg_loss.sum()
        pos_loss = pos_loss.sum()
        num_pos = pos_mask.view(pos_mask.size(0), -1).sum()
        num_neg = neg_mask.view(neg_mask.size(0), -1).sum()

        if num_pos == 0:
            loss = neg_loss
        else:
            loss = pos_loss / num_pos + neg_loss / num_neg
        return loss


class FocalLoss_Ori(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported
    which is proposed in'Focal Loss for Dense Object Detection.
    (https://arxiv.org/abs/1708.02002)' Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for
    well-classified examples (p>0.5) putting more focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param size_average: (bool, optional) By default, the losses are averaged
    over each loss element in the batch.
    """

    def __init__(
        self,
        num_class: int,
        alpha: Any = [0.25, 0.75],
        gamma: int = 2,
        balance_index: int = -1,
        size_average: bool = True,
    ) -> None:
        super(FocalLoss_Ori, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.size_average = size_average
        self.eps = 1e-6

        if isinstance(self.alpha, (list, tuple)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.Tensor(list(self.alpha))
        elif isinstance(self.alpha, (float, int)):
            assert 0 < self.alpha < 1.0, "alpha should be in `(0,1)`)"
            assert balance_index > -1
            alpha = torch.ones((self.num_class))
            alpha *= 1 - self.alpha
            alpha[balance_index] = self.alpha
            self.alpha = alpha
        elif isinstance(self.alpha, torch.Tensor):
            self.alpha = self.alpha
        else:
            raise TypeError(
                "Not support alpha type, expect `int|float|list|tuple|torch.Tensor`"
            )

    def forward(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
            logit = logit.view(-1, logit.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]
        target = target.view(-1, 1)  # [N,d1,d2,...]->[N*d1*d2*...,1]

        # -----------legacy way------------
        #  idx = target.cpu().long()
        # one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        # one_hot_key = one_hot_key.scatter_(1, idx, 1)
        # if one_hot_key.device != logit.device:
        #     one_hot_key = one_hot_key.to(logit.device)
        # pt = (one_hot_key * logit).sum(1) + epsilon

        # ----------memory saving way--------
        pt = logit.gather(1, target).view(-1) + self.eps  # avoid apply
        logpt = pt.log()

        if self.alpha.device != logpt.device:
            alpha = self.alpha.to(logpt.device)
            alpha_class = alpha.gather(0, target.view(-1))
            logpt = alpha_class * logpt
        loss = -1 * torch.pow(torch.sub(1.0, pt), self.gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


def f_score(
    pr: torch.Tensor,
    gt: torch.Tensor,
    beta: int = 1,
    eps: float = 1e-7,
    threshold: Any = None,
    activation: str = "sigmoid",
) -> torch.Tensor:
    """
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    # if activation is None or activation == "none":
    #     activation_fn = lambda x: x
    # elif activation == "sigmoid":
    #     activation_fn = torch.nn.Sigmoid()
    # elif activation == "softmax2d":
    #     activation_fn = torch.nn.Softmax2d()

    # else:
    #     raise NotImplementedError("Activation implemented for sigmoid and softmax2d")

    # pr = activation_fn(pr)
    # gt = torch.unsqueeze(gt, dim=1)

    if pr.dim() > 2:
        # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
        pr = pr.view(pr.size(0), pr.size(1), -1)
        pr = pr.transpose(1, 2).contiguous()  # [N,C,d1*d2..] -> [N,d1*d2..,C]
        pr = pr.view(-1, pr.size(-1))  # [N,d1*d2..,C]-> [N*d1*d2..,C]
    gt = gt.view(-1, 1)  # [N,d1,d2,...]->[N*d1*d2*...,1]

    if threshold is not None:
        pr = (pr > threshold).float()

    tp = torch.sum(gt * pr).to(pr.device)
    fp = (torch.sum(pr) - tp).to(pr.device)
    fn = (torch.sum(gt) - tp).to(pr.device)

    score = ((1 + beta ** 2) * tp + eps) / (
        (1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps
    )
    return score


class DiceLoss(nn.Module):
    def __init__(self, beta: float = 0.5, with_bce: Any = None) -> None:
        super(DiceLoss, self).__init__()
        self.beta = beta
        self.with_bce = with_bce

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.with_bce is not None:
            bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = input * target
        dice = ((1 + self.beta ** 2) * intersection.sum(1) + smooth) / (
            (self.beta ** 2) * target.sum(1) + input.sum(1) + smooth
        )
        dice = 1 - dice.sum() / num

        if self.with_bce is not None:
            dice = dice + self.with_bce * bce
        return dice  # + 0.5 * bce
