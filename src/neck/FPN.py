from turtle import forward
from numpy import insert
from pyrsistent import inc
import torch
import torch.nn as nn


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x2, x1):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)

class FPN(nn.Module):
    def __init__(self,in_c, out_c, scale_factor=2):
        super().__init__()
        
        assert (len(in_c) == 3)
        assert (len(out_c) == 2)
        self.in_c = in_c
        self.out_c = out_c
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c[1]+in_c[2], in_c[1]+in_c[2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_c[1]+in_c[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c[1]+in_c[2], out_c[1], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c[1]),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_c[1]+in_c[0], out_c[1]+in_c[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c[1]+in_c[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c[1]+in_c[0], out_c[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c[0]),
            nn.ReLU(inplace=True)
        )

    def forward(self, feature):
        """
        args
            feature: List(Tensor)  [
                                     b  c1   h      w
                                     b  c2   h/2    w/2
                                     b c3    h/4    w/4
            ]
        return 
            x: Tensor b out_c[0] h w
        """
        assert(len(feature) == 3)
        up = self.up(feature[2])
        x = torch.cat([feature[1], up], dim=1)
        x = self.conv1(x)
        up = self.up(x)
        x = torch.cat([feature[0], up], dim=1)
        return self.conv2(x)

        
