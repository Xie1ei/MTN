import torch
import torch.nn as nn

class VPNTransform(nn.Module):
    def __init__(self, image_featmap_size, space_featmap_size):
        super(VPNTransform, self).__init__()
        ic, ih, iw = image_featmap_size # (256, 16, 16)
        sc, sh, sw = space_featmap_size # (128, 16, 32)
        self.sc = sc
        self.image_featmap_size = image_featmap_size
        self.space_featmap_size = space_featmap_size
        self.fc_transform = nn.Sequential(
                    nn.Linear(ih * iw, sh * sw),
                    nn.SiLU(),
                    nn.Linear(sh * sw, sh * sw),
                    nn.Dropout(p=0.3)
                )
        self.conv1 = nn.Conv2d(ic, sc, kernel_size=1)


    def forward(self, x):
        b,c,h,w = x.shape
        assert h==self.image_featmap_size[1] and w == self.image_featmap_size[2]
        x = x.view(b,c,h*w)
        bev_view = self.fc_transform(x)

        bev_view = bev_view.view(b,c,self.space_featmap_size[1]*4,self.space_featmap_size[2]*4)
        bev_view = self.conv1(bev_view)




        return bev_view