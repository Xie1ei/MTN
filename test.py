

import torch
from yaml import load
from src.backbone.feature_extractor import deepFeatureExtractor_EfficientNet
from src.neck.FPN import FPN, Up
from src.head.det.ObjectBox import Model, C3, Conv

from torchvision.models.resnet import resnet18


from  src.vTrans.FCTransform import Row_block, FCTransform

from src.head import Detect

from src.loss.ObjectCenter import ComputeLoss
from src.dataset.HM_object import ObjectDataset
from torch.utils.data import DataLoader

# --------------------------------------
dataset = ObjectDataset('/root/tools/card/object.mini.txt')
loader  = DataLoader(dataset=dataset, batch_size=10, collate_fn=dataset.collate_fn)
for i, batch in enumerate(loader):
    for k,v in batch.items():
        print(k)
        print(v.shape)
        if k=='object':
            print(v)
    exit()
# print(len(dataset))
# res = dataset[0]
# print(res['object'])

for i in range(4):
    print(i)
exit()

# ---------------------------------------
# model = torch.nn.Linear(2,3)
# loss = ComputeLoss(model)


# pre = [torch.randn(3,1,30,30,5+10) for _ in range(2)]
# target = torch.randn(3,6)
# target[:,1] = 1
# target[:,0] = 1
# ls, oo= loss(pre, target)
# print(ls)
# print(oo)
# exit()

# ------------------------------------

# model = Row_block(3,18, 64, 64, 32)
# model  = FCTransform((3,128,128), (16,100,30))

# ipnut = torch.randn(1,3,128,128)
# out = model(ipnut)
# print(out.shape)
# out : torch.Tensor
# print(out.is_contiguous())

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

backbone = deepFeatureExtractor_EfficientNet(lv3=True,lv4=True,pretrained=False)
# neck = FPN(in_c=[24, 40, 64], out_c=[64, 64])
neck = Up(104, 128)
bevTrans = FCTransform((128,128,256), (64, 120, 30))

trunk = resnet18(pretrained=False, zero_init_residual=True)
bev_neck = trunk.layer1

input = torch.randn(1,3,512,512*2)

out = backbone(input)
for x in out:
    print(x.shape)
out = neck(out[1], out[2])
print(out.shape)
print('--------------To BEV-------------')
bev = bevTrans(out)
print(bev.shape)

out = bev_neck(bev)
print(out.shape)
out = trunk.layer2(out)
print(out.shape)
out = trunk.layer3(out)
print(out.shape)

exit()




    

print(len(out))
for x in out:
    print(x.shape)