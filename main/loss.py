
import torch.nn as nn
import torch
import torch.nn.functional as F


class BinaryMaskLoss(nn.Module):
    def __init__(self, weight=0.8):
        super(BinaryMaskLoss, self).__init__()

        self.weight = weight

    def forward(self, inputs, targets, smooth=1e-6):
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        dice_loss = 1 - dice

        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = 0.8 * (1 - BCE_EXP) ** 2 * BCE
        return self.weight * dice_loss + (1 - self.weight) * focal_loss


class BatchBinaryMaskLoss(nn.Module):
    def __init__(self, weight=0.8, alpha=0.8, gamma=2):
        super(BatchBinaryMaskLoss, self).__init__()
        self.weight = weight
        self.alpha = alpha
        self.gamma = gamma

    def dice_loss(self, inputs, targets, smooth=1e-6):
        inputs = inputs.view(inputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (inputs * targets).sum(dim=1)
        inputs_sum = inputs.sum(dim=1)
        targets_sum = targets.sum(dim=1)

        dice = (2.0 * intersection + smooth) / (inputs_sum + targets_sum + smooth)
        dice_loss = 1 - dice
        return dice_loss.mean()

    def focal_loss(self, inputs, targets):
        BCE = F.binary_cross_entropy(inputs, targets, reduction='none')
        BCE = BCE.view(BCE.size(0), -1).mean(dim=1)

        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1 - BCE_EXP) ** self.gamma * BCE
        return focal_loss.mean()

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)

        # 计算 Dice Loss 和 Focal Loss
        dice_loss = self.dice_loss(inputs, targets)
        focal_loss = self.focal_loss(inputs, targets)

        # 加权总损失
        total_loss = self.weight * dice_loss + (1 - self.weight) * focal_loss
        return total_loss


class BinaryIoU(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BinaryIoU, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth) / (union + smooth)

        return IoU
