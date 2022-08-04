
import torch
import torch.nn as nn



class Row_block(nn.Module):
    def __init__(self, ic,sc, row_size, col_size, re_row):
        super(Row_block, self).__init__()
        """
        Row_block: remap each row for input feature
            
        Args:
            ic : input channels
            sc : output channels
            row_size: weith of input feature
            col_size: height of input feature
            re_row:  weight of output feature
        """
        self.row_size = row_size
        self.col_size = col_size
        self.re_row = re_row
        self.ic = ic

        self.row_conv = nn.ModuleList()
        for i in range(col_size):
            self.row_conv.append(nn.Conv2d(in_channels=ic, out_channels=ic*re_row, kernel_size=(1,row_size),stride=1, groups=ic,bias=False))
        self.channel_fix = nn.Conv2d(ic*re_row,ic*re_row, 1,1,bias=False)
        self.BR1 = nn.Sequential(
           nn.BatchNorm2d(ic),
           nn.ReLU(),
           nn.Conv2d(ic,sc,kernel_size=3, stride=1,padding=1),
           nn.BatchNorm2d(sc),
           nn.ReLU()
        )
        
    def forward(self, x):
        b, c , h , w = x.shape
        assert h == self.col_size
        assert c == self.ic
        res = []

        for i in range(h):
            row_feature = self.row_conv[i](x[:,:,i:i+1,:]) # b c 1 w  --> b c*w 1 1
            res.append(row_feature)
        feature =  torch.cat(res, dim=2)
        feature  = self.channel_fix(feature)
        feature = feature.view(b,c,h, self.re_row) 
        return self.BR1(feature)
        

        

class FCTransform(nn.Module):
    def __init__(self, image_size, space_featmap_size, mode='bilinear'):
        super(FCTransform,self).__init__()
        """"
        ipm 2 BEV  by row FC
        """
        ic, ih, iw = image_size
        sc, sh, sw = space_featmap_size

        self.row_remap = Row_block(ic,sc, iw, ih, int(sw*iw/sh))
        self.space_mapping = nn.Upsample(size=(sh,sw), mode=mode, align_corners=True)

        self.init_module()

    def forward(self, x):
        return self.space_mapping(self.row_remap(x))

    def init_module(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
