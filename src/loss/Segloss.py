import torch
from torch import Tensor, nn as nn


class PushPullLoss(nn.Module):
    """
    An embedding loss to min var of per cluster and max distance between different clusters.

    So, for easier cluster, margin_var should be small, and margin_dist should be larger

    Inputs:
    featmap: prediction of network, [b,1,h,w], float tensor
    gt: gt, [b,1,h,w], long tensor, all val >= ignore_label will NOT be contributed to loss.

    loss = var_weight * var_loss + dist_weight * dist_loss

    Args:
        var_weight (float):
        dist_weight (float):
        margin_var (float): margin for var, any var < this margin will NOT be counted in loss
        margin_dist (float): margin for distance, any distance > this margin will NOT be counted in loss
        ignore_label: val in gt >= this arg, will be ignored.
    """

    def __init__(self, var_weight, dist_weight, margin_var, margin_dist, ignore_label):
        super(PushPullLoss, self).__init__()
        self.var_weight = var_weight
        self.dist_weight = dist_weight
        self.margin_var = margin_var
        self.margin_dist = margin_dist
        self.ignore_label = ignore_label

    def forward(self, featmap, gt):
        assert (featmap.shape == gt.shape)
        pull_loss = []
        push_loss = []
        C = gt[gt < self.ignore_label].max().item()
        # B, 1, H, W
        # TODO not an optimized implement here. Should not expand B dim.
        for b in range(featmap.shape[0]):
            bfeat = featmap[b]
            bgt = gt[b]
            instance_means = {}
            for i in range(1, C + 1):
                instance_mask = bgt == i
                if instance_mask.sum() == 0:
                    continue

                instance_mean = bfeat[instance_mask].mean()
                instance_means[i] = instance_mean
                instance_loss = torch.clip(torch.abs(bfeat[instance_mask] - instance_mean) - self.margin_var,
                                           min=0.0) ** 2
                pull_loss.append(instance_loss.mean())
            for i in range(1, C + 1):
                for j in range(1, C + 1):
                    if i == j:
                        continue  # No need to push
                    if i not in instance_means or j not in instance_means:
                        continue
                    instance_loss = torch.clip(2 * self.margin_dist - torch.abs(instance_means[i] - instance_means[j]),
                                               min=0.0) ** 2
                    push_loss.append(instance_loss)
        if len(pull_loss) > 0:
            pull_loss = torch.cat([item.unsqueeze(0) for item in pull_loss]).mean() * self.var_weight
        else:
            pull_loss = 0.0 * featmap.mean()  # Fake loss

        if len(push_loss) > 0:
            push_loss = torch.cat([item.unsqueeze(0) for item in push_loss]).mean() * self.dist_weight
        else:
            push_loss = 0.0 * featmap.mean()  # Fake loss
        return push_loss + pull_loss




class IoULoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(IoULoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        mask = (targets != self.ignore_index).float()
        targets = targets.float()
        num = torch.sum(outputs * targets * mask)
        den = torch.sum(outputs * mask + targets * mask - outputs * targets * mask)
        return 1 - num / den
