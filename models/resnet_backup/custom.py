# -*- coding=utf-8 -*-

__all__ = [
    'ChannelPruning',
    'SwitchableBatchNorm2d'
]

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

import numpy as np

def get_in_channels(m: nn.Module) -> Optional[int]:
    if isinstance(m, nn.Sequential):
        for sub in m:
            c = get_in_channels(sub)
            if c is not None:
                return c
        return None
    elif isinstance(m, nn.Linear):
        return m.in_features
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return m.in_channels
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        return m.num_features
    else:
        return None


def get_out_channels(m: nn.Module) -> Optional[int]:
    if isinstance(m, nn.Sequential):
        res = None
        for sub in m:
            c = get_in_channels(sub)
            if c is not None:
                res = c
        return res
    elif isinstance(m, nn.Linear):
        return m.out_features
    elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return m.out_channels
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        return m.num_features
    else:
        return None

class ChannelPruning(nn.Module):

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rate = 1.0
        self.loss = None
        self.gate = nn.Linear(in_features=self.in_channels+1,
                              out_features=self.out_channels,
                              bias=True)
        nn.init.constant_(self.gate.bias, 1)
        nn.init.kaiming_normal_(self.gate.weight)

        self.count_channel = np.zeros(self.out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        k = int(self.out_channels * (self.rate))
        s = F.adaptive_avg_pool2d(torch.abs(x), (1, 1)).view(x.size()[0], -1)

        r = torch.FloatTensor(x.size()[0], 1).to(x.device).fill_(self.rate)
        s = torch.cat([s, r], dim=1)

        g = torch.relu(self.gate(s))
        i = (-g).topk(k, 1)[1]
        t = g.scatter(1, i, 0)
        t = t / torch.sum(t, dim=1).unsqueeze(1) * self.out_channels

        mask = t.unsqueeze(2).unsqueeze(3)

        if k == 0:
            self.loss = None
            return mask

        if self.training:
            self.loss = torch.norm(g, 1)
        else:
            self.loss = None
        self.count_channel += np.where(t.cpu().detach().numpy() != 0.0, 1.0, 0.0).sum(axis=0)
        return mask

class SwitchableBatchNorm2d(nn.Module):
    def __init__(self, num_features):
        super(SwitchableBatchNorm2d, self).__init__()
        self.num_features = num_features

    def make_list(self, pruning_rate_list):
        self.pruning_rate_list = pruning_rate_list
        bns = []
        for _ in pruning_rate_list:
            bns.append(nn.BatchNorm2d(self.num_features))
        self.bn = nn.ModuleList(bns)
        self.pruning_rate = max(self.pruning_rate_list)

    def forward(self, input):
        idx = self.pruning_rate_list.index(self.pruning_rate)
        y = self.bn[idx](input)
        return y
