'''MobileNetV3 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init



class hswish(nn.Module):
    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=True) / 6
        return out


class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(in_size // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.BatchNorm2d(in_size),
            hsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.InstanceNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        self.bn2 = nn.InstanceNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.InstanceNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                nn.InstanceNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.bn1(self.conv1(x)))
        out = self.nolinear2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class Block2(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, kernel_size, in_size, expand_size, out_size, nolinear, semodule, stride):
        super(Block2, self).__init__()
        self.stride = stride
        self.se = semodule

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn1 = nn.InstanceNorm2d(expand_size)
        self.nolinear1 = nolinear
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, groups=expand_size, bias=False)
        # self.bn2 = nn.InstanceNorm2d(expand_size)
        self.nolinear2 = nolinear
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn3 = nn.InstanceNorm2d(out_size)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_size != out_size:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0, bias=False),
                # nn.InstanceNorm2d(out_size),
            )

    def forward(self, x):
        out = self.nolinear1(self.conv1(x))
        out = self.nolinear2(self.conv2(out))
        out = self.conv3(out)
        if self.se != None:
            out = self.se(out)
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV3_Large(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_Large, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), None, 1),
            Block(3, 16, 64, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 72, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 72, 40, nn.ReLU(inplace=True), SeModule(40), 2),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(5, 40, 120, 40, nn.ReLU(inplace=True), SeModule(40), 1),
            Block(3, 40, 240, 80, hswish(), None, 2),
            Block(3, 80, 200, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 184, 80, hswish(), None, 1),
            Block(3, 80, 480, 112, hswish(), SeModule(112), 1),
            Block(3, 112, 672, 112, hswish(), SeModule(112), 1),
            Block(5, 112, 672, 160, hswish(), SeModule(160), 1),
            Block(5, 160, 672, 160, hswish(), SeModule(160), 2),
            Block(5, 160, 960, 160, hswish(), SeModule(160), 1),
        )


        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(960, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out



class MobileNetV3_Small(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_Small, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
            Block(5, 24, 96, 40, hswish(), SeModule(40), 2),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 120, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 288, 96, hswish(), SeModule(96), 2),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 1),
        )


        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(576)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(576, 1280)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        out = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.hs3(self.bn3(self.linear3(out)))
        out = self.linear4(out)
        return out

import math
from downsample import Downsample_PASA_group_softmax, Downsample
class MobileNetV3_Mogai(nn.Module):
    def __init__(self, num_classes=1000):
        super(MobileNetV3_Mogai, self).__init__()
        self.ttt = Downsample(channels = 3, stride=2, filt_size=4)
        # self.ttt = nn.UpsamplingBilinear2d(scale_factor=0.5)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Downsample(channels = 16, stride=2, filt_size=3),

            Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 1),
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 1),
            Downsample(channels = 24, stride=2, filt_size=3),
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),

            Block(5, 24, 96, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            # Downsample_PASA_group_softmax(96,3,stride=2),
            # Downsample(channels = 24, stride=2, filt_size=4),
            # nn.UpsamplingBilinear2d(scale_factor=0.5),
        )

        self.modify_c = nn.Sequential(
            Downsample(channels = 40, stride=2, filt_size=3),
            Block(5, 40, 512, 96, hswish(), SeModule(96), 1),
            # Block(5, 96, 288, 96, hswish(), SeModule(96), 1),
            nn.Conv2d(96, 512, kernel_size=1, stride=1, padding=0, bias=False),
            # nn.InstanceNorm2d(512), nooooooooooooooooooo
            # nn.LeakyReLU(negative_slope=0.2),
            # nn.ReLU(True),
            hswish(),
        )

        self.bneck2 = nn.Sequential(
            Block(5, 40, 120, 48, hswish(), SeModule(48), 2),
            Block(5, 48, 144, 48, hswish(), SeModule(48), 1),
            Block(5, 48, 288, 96, hswish(), SeModule(96), 1),
            Block(5, 96, 576, 96, hswish(), SeModule(96), 2),
        )

        # self.upsample = nn.Sequential(
        #     nn.Conv2d(576, 256*64, kernel_size=1, stride=1, padding=0, bias=False),
        #     nn.PixelShuffle(8)
        # )

        self.conv2 = nn.Conv2d(96, 576, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.InstanceNorm2d(576)
        self.hs2 = hswish()
        self.linear3 = nn.Linear(576, 16)
        self.bn3 = nn.LayerNorm(16)
        self.hs3 = hswish()
        self.linear4 = nn.Linear(16, 8)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.2)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        d1 = self.ttt(x)

        out = self.conv1(d1)
        out = self.bn1(out)
        out = self.hs1(out)
        #scale = math.sqrt(2)* (1 / math.sqrt(16 * 3 ** 2))
        #out = out1*scale

        out = self.bneck(out)
        #scale = math.sqrt(2)* (1 / math.sqrt(40 * 1 ** 2))
        #out = out*scale

        content = self.modify_c(out)
        scale = math.sqrt(2)* (1 / math.sqrt(40 * 1 ** 2))
        content = content*scale

        out = self.bneck2(out)
        out = self.hs2(self.bn2(self.conv2(out)))
        # print(out.shape)
        out = F.avg_pool2d(out, 8)
        # print(out.shape)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.hs3(self.bn3(self.linear3(out)))
        style = self.linear4(out)
        # style = torch.randn(1, 8)
        
        return content, style

# from thop import profile
# import sys
# sys.path.append('../GANsNRoses/')
# from model import Encoder
# from matplotlib import pyplot as plt 
# import numpy as np
# from skimage import io
# from skimage import transform

# def test():
#     torch.set_printoptions(precision=4, sci_mode=False)
#     encoder = Encoder(256, 8, 4, 1, 1)
#     net = MobileNetV3_Mogai()

#     # pretext_model = torch.load("mbv3_small.pth.tar", map_location='cpu')
#     # weight = pretext_model["state_dict"]
#     # #print (pretext_model)
#     # model2_dict = net.state_dict()#初始化一个新的参数字典，key是辛模型的
#     # state_dict = {k:v for k,v in weight.items() if k in model2_dict.keys()}#把未改变的key和value找出来
#     # print (weight)
#     # model2_dict.update(state_dict)#更新到新参数字典中
#     # net.load_state_dict(model2_dict)#并载入新的模型里

#     img = io.imread("/Users/cr/git/face/Morph-UGATIT/datasets/xinggan_face/150830_0.jpg")
#     img = transform.resize(img, (256, 256))
#     img_tensor = torch.from_numpy(img.transpose((2,0,1)))
#     x2 = torch.reshape(img_tensor, (1, 3, 256, 256)).float()
#     # x2 = img_tensor.repeat((1, 256, 1, 1)).float()

#     x = torch.randn(2,3,256,256)
#     content, style = net(x2)
#     content2, style2 = encoder(x2)
#     print(content.shape)
#     print(style.shape)
#     print(content)
#     print("-----------------")
#     print(content2)

#     million = 100 * 10000
#     flops256, _ = profile(net, (x,))
#     print("encoder flops", flops256/million)

#     out = torch.reshape(content[0,0,:,:], (16, 16))
#     out = out.detach().numpy()
#     # out = out - np.min(out)/(np.max(out)-np.min(out))
#     out2 = torch.reshape(content2[0,0,:,:], (16, 16))
#     out2 = out2.detach().numpy()
#     # out2 = out2 - np.min(out2)/(np.max(out2)-np.min(out2))

#     # d1 = torch.reshape(d1[0,4,:,:], (128, 128))
#     # d1 = d1.detach().numpy()
#     # # d1 = d1.transpose((1,2,0))

#     # kkk = torch.reshape(kkk[0,4,:,:], (256, 256))
#     # kkk = kkk.detach().numpy()

#     plt.figure("haha")
#     plt.subplot(1,2,1), plt.title('a')
#     plt.imshow(out)
#     plt.subplot(1,2,2), plt.title('b')
#     plt.imshow(out2)

#     # plt.figure("kkk")
#     # plt.subplot(1,3,1), plt.title('a')
#     # plt.imshow(d1)
#     # plt.subplot(1,3,2), plt.title('b')
#     # plt.imshow(kkk)
#     # plt.subplot(1,3,3), plt.title('c')
#     # plt.imshow(img)

#     plt.show()

# test()
