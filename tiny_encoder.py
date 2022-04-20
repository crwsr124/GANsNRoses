'''MobileNetV3 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math


# replace_relu6 = relu.clamp(max=6)
# def _relu6(self,x):
#     return nn.ReLU6(x)
# def _hard_swish(self,x):
#     return x*_relu6(x+3.0)/6.0
class hswish(nn.Module):
    def __init__(self):
        super(hswish, self).__init__()
        self.ac = nn.Hardswish()
    def forward(self, x):
        # out = x * F.relu6(x + 3, inplace=True) / 6
        # out = x * F.relu(x + 3, inplace=True).clamp(max=6) / 6
        out = self.ac(x)
        return out

class hsigmoid(nn.Module):
    def __init__(self):
        super(hsigmoid, self).__init__()
        self.hs = nn.Hardsigmoid()
    def forward(self, x):
        out = self.hs(x)
        return out

# class hswish(nn.Module):
#     def forward(self, x):
#         out = x * F.relu6(x + 3, inplace=True) / 6
#         return out

# class hsigmoid(nn.Module):
#     def __init__(self):
#         super(hsigmoid, self).__init__()
#         self.relu6 = nn.ReLU6()
#     def forward(self, x):
#         # out = F.relu6(x + 3, inplace=True) / 6
#         out = self.relu6(x + 3) / 6
#         return out


class SeModule(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, in_size // reduction, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(in_size // reduction),
            nn.LayerNorm([in_size // reduction, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_size // reduction, in_size, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.BatchNorm2d(in_size),
            nn.LayerNorm([in_size, 1, 1]),
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

class TinyEncoder_origin(nn.Module):
    def __init__(self, num_classes=1000):
        super(TinyEncoder_origin, self).__init__()
        # self.ttt = Downsample(channels = 3, stride=2, filt_size=4)
        self.ttt = nn.UpsamplingBilinear2d(scale_factor=0.5)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(16)
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            # Downsample(channels = 16, stride=2, filt_size=3),

            Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 1),
            Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            # Downsample(channels = 24, stride=2, filt_size=3),
            Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),

            Block(5, 24, 96, 40, hswish(), SeModule(40), 1),
            Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
            # Downsample_PASA_group_softmax(96,3,stride=2),
            # Downsample(channels = 24, stride=2, filt_size=4),
            # nn.UpsamplingBilinear2d(scale_factor=0.5),
        )

        self.modify_c = nn.Sequential(
            # Downsample(channels = 40, stride=2, filt_size=3),
            Block(5, 40, 512, 96, hswish(), SeModule(96), 2),
            # Block(5, 96, 288, 96, hswish(), SeModule(96), 1),
            nn.Conv2d(96, 512, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.InstanceNorm2d(512), nooooooooooooooooooo
            nn.LeakyReLU(negative_slope=0.2),
            # nn.ReLU(True),
            # hswish(),
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

        out = self.bneck(out)

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

class TinyEncoder_tt(nn.Module):
    def __init__(self):
        super(TinyEncoder_tt, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.InstanceNorm2d(16)         # bias can be false when using Batch Norm OR Instance Norm
        self.hs1 = hswish()

        self.bneck = nn.Sequential(
            Block(3, 16, 32, 32, nn.LeakyReLU(negative_slope=0.2), SeModule(32), 2),
            Block(3, 32, 64, 32, nn.LeakyReLU(negative_slope=0.2), None, 2),
            Block(3, 32, 96, 32, nn.LeakyReLU(negative_slope=0.2), None, 1),

            # Block(3, 16, 16, 16, nn.ReLU(inplace=True), SeModule(16), 2),
            # Block(3, 16, 72, 24, nn.ReLU(inplace=True), None, 2),
            # Block(3, 24, 88, 24, nn.ReLU(inplace=True), None, 1),
            # Block(5, 24, 96, 40, hswish(), SeModule(40), 1),
            # Block(5, 40, 240, 40, hswish(), SeModule(40), 1),
        )

        self.modify_c = nn.Sequential(
            Block(5, 32, 128, 64, hswish(), SeModule(64), 2),
            Block(5, 64, 256, 128, hswish(), None, 1),
            nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0, bias=True),
            # nn.InstanceNorm2d(512), nooooooooooooooooooo
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.bneck2 = nn.Sequential(
            Block(5, 32, 120, 48, hswish(), SeModule(48), 2),
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
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.hs1(out)

        out = self.bneck(out)

        content = self.modify_c(out)
        scale = math.sqrt(2)* (1 / math.sqrt(128 * 1 ** 2))
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

    # def forward(self, x):
    #     d1 = self.ttt(x)

    #     out = self.conv1(d1)
    #     out = self.bn1(out)
    #     out = self.hs1(out)

    #     out = self.bneck(out)

    #     content = self.modify_c(out)
    #     scale = math.sqrt(2)* (1 / math.sqrt(40 * 1 ** 2))
    #     content = content*scale
        
    #     return content

class ScaledConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True, groups=1
    ):
        super().__init__()
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        # self.conv = nn.utils.weight_norm(nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=groups))
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)
        # init.kaiming_normal_(self.conv.weight, mode='fan_out')
        init.normal_(self.conv.weight)
        if self.conv.bias is not None:
            init.constant_(self.conv.bias, 0)

    def forward(self, input):
        out = self.conv(input)*self.scale
        return out

class ScaledLeakyReLU(nn.Module):
    def __init__(
        self, negative_slope=0.2
    ):
        super().__init__()
        self.lrelu = nn.LeakyReLU(negative_slope=negative_slope)

    def forward(self, input):
        out = self.lrelu(input)*math.sqrt(2)
        return out

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, downsample=True):
        super().__init__()

        stride=1
        if downsample:
            stride = 2

        self.conv1 = ScaledConv2d(in_channel, in_channel, 3, padding=1)
        self.lrelu1 = ScaledLeakyReLU()
        self.conv2 = ScaledConv2d(in_channel, out_channel, 3, padding=1, stride=stride)
        self.lrelu2 = ScaledLeakyReLU()

        if downsample:
            self.down = nn.UpsamplingBilinear2d(scale_factor = 0.5)
        else:
            self.down = None

        if in_channel != out_channel:
            self.skip = ScaledConv2d(in_channel, out_channel, 1)
        else:
            self.skip = None

    def forward(self, input):
        out = self.lrelu1(self.conv1(input))
        out = self.lrelu2(self.conv2(out))

        if self.down is None:
            input = input
        else:
            input = self.down(input)

        if self.skip is None:
            skip = input
        else:
            skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)
        return out

class TinyEncoder_large(nn.Module):
    def __init__(self):
        super(TinyEncoder_large, self).__init__()
        self.conv1 = ScaledConv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  #256x256
        self.hs1 = ScaledLeakyReLU()

        
        self.bneck = nn.Sequential(
            ResBlock(64, 128, downsample=True),      #128x128
            ResBlock(128, 256, downsample=True),     #64x64
            ResBlock(256, 512, downsample=True),     #32x32
            ResBlock(512, 512, downsample=True),     #16x16

            ResBlock(512, 512, downsample=False),
            ResBlock(512, 512, downsample=False),
        )

        self.content = nn.Sequential(
                ScaledConv2d(512, 512, 1), 
                ScaledLeakyReLU(),
                ScaledConv2d(512, 512, 1),
                ScaledLeakyReLU(),
        )

        self.linear = nn.Sequential(
            nn.AvgPool2d(16),
            nn.Flatten(),
            nn.Linear(512, 8)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.hs1(out)

        out = self.bneck(out)

        content = self.content(out)

        style = self.linear(content)
        return content, style



class ResBlock_s(nn.Module):
    def __init__(self, in_channel, expand_channel, out_channel, downsample=True, semodule = False):
        super().__init__()

        stride=1
        if downsample:
            stride = 2

        self.conv1 = ScaledConv2d(in_channel, expand_channel, 1)
        self.lrelu1 = ScaledLeakyReLU()
        self.conv2 = ScaledConv2d(expand_channel, expand_channel, 3, padding=1, stride=stride, groups=expand_channel)
        self.lrelu2 = ScaledLeakyReLU()
        self.conv3 = ScaledConv2d(expand_channel, out_channel, 1)

        if semodule:
            self.se = SeModule(out_channel)
        else:
            self.se = None

        if downsample:
            self.down = nn.UpsamplingBilinear2d(scale_factor = 0.5)
        else:
            self.down = None

        if in_channel != out_channel:
            self.skip = ScaledConv2d(in_channel, out_channel, 1)
        else:
            self.skip = None

    def forward(self, input):
        out = self.lrelu1(self.conv1(input))
        out = self.lrelu2(self.conv2(out))
        out = self.conv3(out)
        if self.se != None:
            out = self.se(out)

        if self.down is None:
            input = input
        else:
            input = self.down(input)

        if self.skip is None:
            skip = input
        else:
            skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)
        return out

class TinyEncoder(nn.Module):
    def __init__(self):
        super(TinyEncoder, self).__init__()
        self.conv1 = ScaledConv2d(3, 16, kernel_size=3, stride=2, padding=1)  #128x128
        self.hs1 = ScaledLeakyReLU()
        
        self.bneck = nn.Sequential(
            ResBlock_s(16, 16, 16, downsample=True, semodule=True),      #128x128
            ResBlock_s(16, 64, 32, downsample=True),     #64x64
            ResBlock_s(32, 128, 64, downsample=False),     #64x64
            ResBlock_s(64, 256, 128, downsample=True, semodule=True),     #32x32
            ResBlock_s(128, 512, 256, downsample=False, semodule=True),     #16x16
        )

        self.content = nn.Sequential(
            ScaledConv2d(256, 512, 1), 
            ScaledLeakyReLU(),
        )

        self.linear = nn.Sequential(
            nn.AvgPool2d(16),
            nn.Flatten(),
            nn.Linear(512, 8)
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.hs1(out)

        out = self.bneck(out)

        content = self.content(out)

        style = self.linear(content)
        return content, style


from matplotlib import pyplot as plt 
from skimage import io
from torchvision import transforms
from thop import profile

totensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=True)
    ])

def test():
    net = TinyEncoder()

    img = io.imread("/Users/cr/git/face/Morph-UGATIT/datasets/xinggan_face/150830_0.jpg")
    x_img = totensor(img)
    x_img = torch.reshape(x_img, (1, 3, 256, 256))
    # print(torch.min(x_img))
    # print(torch.max(x_img))
    x_rand = torch.randn(1,3,256,256)

    content, style = net(x_img)
    content_rand, style = net(x_rand)
    # print(content.shape)
    # print(style.shape)
    # kkkkk = WNConv2d(3, 1, 3)(x_img)
    # print("111111111111", kkkkk)
    # ttttt = EqualConv2d(3, 1, 3)(x_img)
    # print("222222222222", ttttt)

    million = 100 * 10000
    flops256, _ = profile(net, (x_img,))
    print("decoder flops:", flops256/million)

    x_img = torch.reshape(x_img, (x_img.shape[1], x_img.shape[2], x_img.shape[3]))
    x_img = x_img.detach().numpy().transpose((1, 2, 0))
    x_img = (x_img+1.0)/2.0
    x_rand = torch.reshape(x_rand, (x_rand.shape[1], x_rand.shape[2], x_rand.shape[3]))
    x_rand = x_rand.detach().numpy().transpose((1, 2, 0))
    x_rand = (x_rand+1.0)/2.0
    content = content[0, 128, :, :].detach().numpy()
    content_rand = content_rand[0, 128, :, :].detach().numpy()

    plt.figure("encoder")
    plt.subplot(2,2,1), plt.title('x_img')
    plt.imshow(x_img)
    plt.subplot(2,2,2), plt.title('content')
    plt.imshow(content)
    plt.subplot(2,2,3), plt.title('x_rand')
    plt.imshow(x_rand)
    plt.subplot(2,2,4), plt.title('content_rand')
    plt.imshow(content_rand)
    plt.show()

import onnxruntime as rt
import time
import numpy as np
def toonnx():

    net = TinyEncoder()
    
    x_rand = torch.randn(1,3,256,256)
    content, _ = net(x_rand)

    export_onnx_file = "encoder.onnx"
    torch.onnx.export(
        net,
        (x_rand),
        export_onnx_file,
        opset_version=11,
        do_constant_folding=True,
        input_names=["x"],
        output_names=["content", "style"],
        training = torch.onnx.TrainingMode.EVAL,
        # verbose=True,
        # operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
        # dynamic_axes={"input":{0:"batch_size",2:"batch_size",3:"batch_size"}, "output":{0:"batch_size"}}
        )

    rand_x_in = x_rand.detach().cpu().numpy()
    rand_x_in = rand_x_in.astype(np.float32)
    sess = rt.InferenceSession(export_onnx_file)

    start_time = time.time()
    out = sess.run([], {
            'x': rand_x_in
        })
    elapse_time = time.time() - start_time
    print(elapse_time)
    
    content = content[0, 128, :, :].detach().numpy()
    out = out[0][0, 128, :, :]

    plt.figure("encoder")
    plt.subplot(1,2,1), plt.title('origin')
    plt.imshow(content)
    plt.subplot(1,2,2), plt.title('onnx')
    plt.imshow(out)
    plt.show()

# toonnx()
# test()
