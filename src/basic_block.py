import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, module, downsample=None):
        super(Residual, self).__init__()
        self.module = module
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.module(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class DoubleCBR(nn.Module):
    def __init__(self, ic, mc, oc, dropout=0.2):
        super(DoubleCBR, self).__init__()
        self.mod = nn.Sequential(
            nn.Conv2d(ic, mc, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mc),
            nn.ReLU(),

            nn.Dropout2d(p=dropout),

            nn.Conv2d(mc, oc, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(oc),
            # nn.ReLU(),
        )

    def forward(self, x):
        return self.mod(x)
