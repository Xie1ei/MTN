
from os import fchdir
import torch
from src.backbone.feature_extractor import deepFeatureExtractor_EfficientNet
from src.neck.FPN import FPN
from src.head.det.ObjectBox import Model, C3, Conv

from torchvision.models.resnet import resnet18


from  src.vTrans.FCTransform import Row_block, FCTransform

# model = Row_block(3,18, 64, 64, 32)
model  = FCTransform((3,128,128), (16,100,30))

ipnut = torch.randn(1,3,128,128)
out = model(ipnut)
print(out.shape)
out : torch.Tensor
print(out.is_contiguous())

# model = torch.nn.Conv2d(3,9,1,1,0,groups=3)
# input = torch.randn(1,3,128,128)
# print(model.weight.shape)

# model = resnet18(pretrained=False, zero_init_residual=True)
# print(model.layer1)
# print(model.layer2)
# print(model.layer3)
# input = torch.randn(1,64,128,128)
# out = model.layer1(input)
# print(out.shape)
# out = model.layer2(out)
# print(out.shape)
# out = model.layer3(out)
# print(out.shape)

# model = C3(32, 64)
# model = Conv(32,64,k=3,s=2,p=1)
# input = torch.randn(1,32,128,128)
# out = model(input)
# print(out.shape)

# model = Model(cfg='configs/objectBox.yaml',ch=3,nc=5)
# print(model)
# input = torch.randn(1,3,512,512)
# out = model(input)
# for i in out:
#     print(i.shape)
# exit()

# backbone = deepFeatureExtractor_EfficientNet(lv3=True,lv4=True,pretrained=False)
# neck = FPN(in_c=[24, 40, 64], out_c=[64, 64])

# input = torch.randn(1,3,512,512)

# out = backbone(input)
# for x in out:
#     print(x.shape)
# out = neck(out)
# print(out.shape)
exit()




    

print(len(out))
for x in out:
    print(x.shape)