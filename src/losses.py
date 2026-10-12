import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.5, focal_weight=0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def focal_loss(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        return (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

    def dice_loss(self, inputs, targets):
        smooth = 1.0
        probs = F.softmax(inputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=2).float()
        prob_class1 = probs[:, 1]
        target_class1 = targets_one_hot[:, 1]
        intersection = (prob_class1 * target_class1).sum()
        union = prob_class1.sum() + target_class1.sum() + smooth
        return 1 - (2.0 * intersection) / union

    def forward(self, inputs, targets):
        return self.focal_weight * self.focal_loss(inputs, targets) + self.dice_weight * self.dice_loss(inputs, targets)