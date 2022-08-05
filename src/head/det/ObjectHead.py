
import torch
import torch.nn as nn
from src.utils.Object_utils import xyxy2xywh, make_divisible
from src.utils.plot import feature_visualization


class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, det_layers=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per detection layers
        self.nl = len(det_layers)  # number of detection layers
        self.na = 1

        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(det_layers).float().view(self.nl, -1, 2)[:, :self.na, :]
        self.register_buffer('det_layers', a)  # shape(nl,na,2)
        self.register_buffer('det_layers_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)


    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4] or self.onnx_dynamic:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)

                y = x[i].sigmoid()

                coord = torch.ones([bs, self.na, ny, nx], device=y.device)
                coord = coord.nonzero(as_tuple=False) 
                x_coord = coord[:, 3]
                y_coord = coord[:, 2]
                
                power = 2 ** i

                s_gain = torch.ones_like(self.det_layers_grid[i, ..., 0]) * power
                dx1 = (y[..., 0] * 2) ** 2 * s_gain
                dy1 = (y[..., 1] * 2) ** 2 * s_gain
                dx2 = (y[..., 2] * 2) ** 2 * s_gain
                dy2 = (y[..., 3] * 2) ** 2 * s_gain
                
                y[..., 0] = (x_coord.view(bs, self.na, ny, nx)+1 - (dx1)) * self.stride[i]
                y[..., 1] = (y_coord.view(bs, self.na, ny, nx)+1 - (dy1)) * self.stride[i]
                y[..., 2] = (x_coord.view(bs, self.na, ny, nx) + (dx2)) * self.stride[i]
                y[..., 3] = (y_coord.view(bs, self.na, ny, nx) + (dy2)) * self.stride[i]

                xyxy = y[..., :4].view(-1,4)
                xywh = xyxy2xywh(xyxy)
                y[..., :4] = xywh.view(bs, self.na, ny, nx, 4)

                pred = y.view(bs, -1, self.no)
                z.append(pred)

                                
        return x if self.training else (torch.cat(z, 1), x)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()