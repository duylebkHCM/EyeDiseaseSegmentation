from typing import Optional, Dict

import torch
import numpy as np
import torch.nn.functional as F
from pytorch_toolbelt.losses import *
from torch import nn
from torch.nn import KLDivLoss
from .lovasz import lovasz_hinge, BinaryLovaszLoss as CustomLovaszLoss

__all__ = ["get_loss", "WeightedBCEWithLogits", "KLDivLossWithLogits"]


class BinaryKLDivLossWithLogits(KLDivLoss):
    """
    """

    def __init__(self, ignore_index=None):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        # Resize target to size of input
        input_size = input.size()[2:]
        target_size = target.size()[2:]
        if input_size != target_size:
            if self.ignore_index is not None:
                raise ValueError("In case ignore_index is not None, input and output tensors must have equal size")
            target = F.interpolate(target, size=input_size, mode="bilinear", align_corners=False)

        if self.ignore_index is not None:
            mask = target != self.ignore_index
            input = input[mask]
            target = target[mask]

            if len(target) == 0:
                return 0

        input = torch.cat([input, 1 - input], dim=1)
        log_p = F.logsigmoid(input)

        target = torch.cat([target, 1 - target], dim=1)

        loss = F.kl_div(log_p, target, reduction="mean")
        return loss


class ResizePredictionTarget2d(nn.Module):
    """
    Wrapper around loss, that rescale model output to target size
    """

    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def forward(self, input, target):
        input = F.interpolate(input, target.size()[2:], mode="bilinear", align_corners=False)
        return self.loss(input, target)


class ResizeTargetToPrediction2d(nn.Module):
    """
    Wrapper around loss, that rescale target tensor to the size of output of the model.
    Note: This will corrupt binary labels and not indended for multiclass case
    """

    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def forward(self, input, target):
        target = F.interpolate(target, input.size()[2:], mode="bilinear", align_corners=False)
        return self.loss(input, target)



class WeightedBCEWithLogits(nn.Module):
    def __init__(self, pos_weights, ignore_index: Optional[int] = -100, reduction="mean"):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.pos_weights = pos_weights

    def forward(self, label_input: torch.Tensor, target: torch.Tensor):

        if self.ignore_index is not None:
            not_ignored_mask = (target != self.ignore_index).float()

        loss = nn.BCEWithLogitsLoss(reduce=None, pos_weight=self.pos_weights)(label_input, target)

        if self.ignore_index is not None:
            loss = loss * not_ignored_mask.float()

        if self.reduction == "mean":
            loss = loss.mean()

        if self.reduction == "sum":
            loss = loss.sum()

        return loss

class TopKLoss(nn.Module):
    def __init__(self, ignore_index: Optional[int] = -100, reduction="mean", topk=10):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.topk = topk
        
    def forward(self, label_input: torch.Tensor, target: torch.Tensor):
        if self.ignore_index is not None:   
            not_ignored_mask = (target != self.ignore_index).float()

        copy_target = target.int()
        fore_ground = copy_target.detach().cpu() == 1
        back_ground = copy_target.detach().cpu() == 0

        fg_loss = nn.BCEWithLogitsLoss(reduction='none')(label_input[fore_ground], target[fore_ground])
        bg_loss = nn.BCEWithLogitsLoss(reduction='none')(label_input[back_ground], target[back_ground])
        topkloss, _ = torch.topk(bg_loss, int(torch.sum(fore_ground)), largest=True, sorted=False)
        # beta = int(torch.sum(back_ground)*self.topk / 100) / (int(torch.sum(back_ground)*self.topk / 100) + torch.sum(fore_ground))
        beta=1/2
        if self.reduction == "mean":
            loss = beta*fg_loss.mean() + (1-beta)*topkloss.mean()

        if self.reduction == "sum":
            loss = beta*fg_loss.sum() + (1-beta)*topkloss.sum()

        if self.ignore_index is not None:
            loss = loss * not_ignored_mask.float()

        return loss
    
class KLDivLossWithLogits(KLDivLoss):
    """
    """

    def __init__(self):
        super().__init__()

    def forward(self, input, target):

        # Resize target to size of input
        target = F.interpolate(target, size=input.size()[2:], mode="bilinear", align_corners=False)

        input = torch.cat([input, 1 - input], dim=1)
        log_p = F.logsigmoid(input)

        target = torch.cat([target, 1 - target], dim=1)

        loss = F.kl_div(log_p, target, reduction="mean")
        return loss

class SymmetricLovasz(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, outputs, targets):
        return 0.5*(lovasz_hinge(outputs, targets) + lovasz_hinge(-outputs, 1.0 - targets))

class LogBCE(nn.Module):
    def __init__(self, ignore_index=None, reduction='mean', smooth_factor=0.1):
        super(LogBCE, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor

    def forward(self, inputs, targets):
        with torch.no_grad():
            beta = targets.mean(dim=[2, 3], keepdims=True)

        if self.smooth_factor is not None:
            soft_targets = ((1 - targets) * self.smooth_factor + targets * (1 - self.smooth_factor)).type_as(inputs)
        else:
            soft_targets = targets.type_as(inputs)

        logits_1 = F.logsigmoid(inputs)
        logits_2 = F.logsigmoid(-inputs)
        loss = -(1-beta)*logits_1*soft_targets - beta*logits_2*(1-soft_targets)

        if self.ignore_index is not None:
            not_ignored_mask = targets != self.ignore_index
            loss *= not_ignored_mask.type_as(loss)

        if self.reduction == "mean":
            loss = loss.mean()

        if self.reduction == "sum":
            loss = loss.sum()

        return loss

def get_loss(loss_name: str, ignore_index=None):
    if loss_name.lower() == "kl":
        return KLDivLossWithLogits()
    
    if loss_name.lower() == "topk":
        return TopKLoss(ignore_index=ignore_index)

    if loss_name.lower() == "bce":
        return SoftBCEWithLogitsLoss(ignore_index=ignore_index)

    if loss_name.lower() == 'wbce':
        return WeightedBCEWithLogits(ignore_index=ignore_index)

    if loss_name.lower() == 'log_bce':
        return LogBCE(ignore_index=ignore_index)

    if loss_name.lower() == "ce":
        return nn.CrossEntropyLoss()

    if loss_name.lower() == "soft_bce":
        return SoftBCEWithLogitsLoss(smooth_factor=0.1, ignore_index=ignore_index)

    if loss_name.lower() == "focal":
        return BinaryFocalLoss(alpha=None, gamma=1.5, ignore_index=ignore_index)

    if loss_name.lower() == "jaccard":
        assert ignore_index is None
        return JaccardLoss(mode="binary")

    if loss_name.lower() == "lovasz":
        assert ignore_index is None
        return CustomLovaszLoss()
    
    if loss_name.lower() == "symmetric_lovasz":
        return SymmetricLovasz()

    if loss_name.lower() == "log_jaccard":
        assert ignore_index is None
        return JaccardLoss(mode="binary", log_loss=True)

    if loss_name.lower() == "dice":
        assert ignore_index is None
        return DiceLoss(mode="binary", log_loss=False)

    if loss_name.lower() == "log_dice":
        assert ignore_index is None
        return DiceLoss(mode="binary", log_loss=True)

    raise KeyError(loss_name)
