from math import factorial
import torch
import torch.nn as nn
from torch.nn.modules.padding import ReflectionPad2d
from torch.nn.modules.pooling import FractionalMaxPool2d
from torch.nn.parameter import Parameter
import math


class PixelNorm(nn.Module):
    def __init__(self, num_channels=None):
        super().__init__()
        # num_channels is only used to match function signature with other normalization layers
        # it has no actual use

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-5)

class MLP(nn.Module):
    def __init__(self, inc, dim, n_layers):
        super().__init__()
        ActFunc = nn.LeakyReLU(0.2)
        mlp = [PixelNorm(),
               nn.Linear(inc, dim),
               ActFunc,
               PixelNorm()]
        for i in range(n_layers-2):
            mlp.extend([
                nn.Linear(dim, dim),
                ActFunc,
                PixelNorm()
            ])
        mlp.extend(
            [nn.Linear(dim, dim),
             PixelNorm()])
        self.dim = dim
        self.mlp = nn.Sequential(*mlp)


    def forward(self, x):
        b, c = x.size(0), x.size(1)
        x = x.view(b, c)
        x = self.mlp(x)
        return x

class LayerInstanceNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.eps = 1e-6
        self.gamma = nn.Parameter(torch.FloatTensor(1, dim, 1, 1).fill_(1.0))
        self.beta = nn.Parameter(torch.FloatTensor(1, dim, 1, 1).fill_(0.0))
        self.rho = nn.Parameter(torch.FloatTensor(1, dim, 1, 1).fill_(0.0))

    def forward(self, x):
        b, c, h, w = x.shape
        ins_mean = x.view(b, c, -1).mean(dim=2).view(b, c, 1, 1)
        ins_val = x.view(b, c, -1).var(dim=2).view(b, c, 1, 1) + self.eps
        ins_std = ins_val.sqrt()

        ln_mean = x.view(b, -1).mean(dim=1).view(b, 1, 1, 1)
        ln_val = x.view(b, -1).var(dim=1).view(b, 1, 1, 1) + self.eps
        ln_std = ln_val.sqrt()

        rho = torch.clamp(self.rho, 0, 1)
        x_ins = (x - ins_mean) / ins_std
        x_ln = (x - ln_mean) / ln_std

        x_hat = rho * x_ins + (1 - rho) * x_ln
        return x_hat * self.gamma + self.beta

class AdaLIN(nn.Module):
    def __init__(self, z_dim=256):
        super().__init__()
        self.eps = 1e-6
        self.rho = nn.Parameter(torch.FloatTensor(1).fill_(1.0))
        self.gamma = nn.Linear(z_dim, z_dim)
        self.beta = nn.Linear(z_dim, z_dim)
    def forward(self, x, z):
        b,c,h,w = x.shape
        ins_mean = x.view(b,c, -1).mean(dim=2).view(b, c, 1, 1)
        ins_var = x.view(b,c,-1).var(dim=2) + self.eps
        ins_std = ins_var.sqrt().view(b, c, 1, 1)

        x_ins = (x - ins_mean) / ins_std

        ln_mean = x.view(b, -1).mean(dim=1).view(b, 1, 1, 1)
        ln_val = x.view(b, -1).var(dim=1).view(b, 1, 1, 1) + self.eps
        ln_std = ln_val.sqrt()

        x_ln = (x - ln_mean) / ln_std

        rho = (self.rho - 0.1).clamp(0, 1.0)  # smoothing
        x_hat = rho * x_ins + (1-rho) * x_ln

        gamma = self.gamma(z).view(b, c, 1, 1)
        beta = self.beta(z).view(b, c, 1, 1)
        # print("::::::::sss")
        # print(gamma)
        # print(beta)

        x_hat = x_hat * gamma + beta
        # x_hat = x_hat * 1.5 + 5
        return x_hat

class ResBlockByAdaLIN(nn.Module):
    def __init__(self, dim):
        super().__init__()

        fan_in = dim * 3 ** 2
        self.scale = 1 / math.sqrt(fan_in)

        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 1, 0, groups=dim//4),
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3, 1, 0, groups=dim//4),
        )
        self.addin_1 = AdaLIN(dim)
        self.addin_2 = AdaLIN(dim)
        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU(0.2, True)

    def forward(self, x, z):
        x1 = self.conv1(x)
        x1 = self.relu(self.addin_1(x1, z))

        x2 = self.conv2(x1)
        x2 = self.addin_2(x2, z)
        return x + x2

class CRGenerator_small(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=6, img_size=256, light=False):
        assert(n_blocks >= 0)
        super(CRGenerator_small, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        self.encoder = CREncoder2()
        self.decoder = CRDecoder()

    def forward(self, input, z_noise):
        encoder_out, feature, z, cam_logit, heatmap = self.encoder(input)
        z = z + z_noise
        out = self.decoder(encoder_out, z)
        return out, cam_logit, heatmap, encoder_out, feature, z

class CREncoder(nn.Module):
    def __init__(self):
        # assert(n_blocks >= 0)
        super(CREncoder, self).__init__()
        self.input_nc = 3
        self.output_nc = 256
        # self.ngf = 64
        self.n_blocks = 6
        self.img_size = 256
        self.light = False

        # in:256x256 out:256x256
        self.DownBlock1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(self.input_nc, 64, kernel_size=7, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True))
        # in:256x256 out:128x128
        self.DownBlock2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True))
        # in:128x128 out:64x64
        self.DownBlock3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True))

        # in:64x64 out:32x32
        self.DownBlock4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True))

        # in:32x32 out:16x16
        self.DownBlock5 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True))

        # in:16x16 out:8x8
        self.DownBlock6 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True))

        # Down-Sampling Bottleneck
        self.ResBlock = nn.Sequential(
            ResnetBlock(256, use_bias=False),
            ResnetBlock(256, use_bias=False),
            ResnetBlock(256, use_bias=False))
        
        # Class Activation Map
        self.gap_fc = nn.Linear(256, 1, bias=False)
        self.gmp_fc = nn.Linear(256, 1, bias=False)
        self.conv1x1 = nn.Conv2d(256 * 2, 256, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(True)

        # z generator
        self.FC = nn.Sequential(
            nn.Linear(256*8*8, 256, bias=False),
            nn.ReLU(True),
            nn.Linear(256, 256, bias=False))

    def forward(self, input):
        # downsample
        x = self.DownBlock1(input)
        #print("1: " + str(x.shape))
        x = self.DownBlock2(x)
        #print("2: " + str(x.shape))
        encoder_out = self.DownBlock3(x)
        x = self.DownBlock4(encoder_out)
        #print("4: " + str(x.shape))
        x = self.DownBlock5(x)
        #print("5: " + str(x.shape))
        x = self.DownBlock6(x)
        #print("6: " + str(x.shape))

        # CAM
        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = encoder_out * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = encoder_out * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        encoder_out = torch.cat([gap, gmp], 1)
        encoder_out = self.relu(self.conv1x1(encoder_out))
        #print("7encoder_out: " + str(encoder_out.shape))
        heatmap = torch.sum(encoder_out, dim=1, keepdim=True)
        #print("8: " + str(heatmap.shape))

        # z
        feature = self.ResBlock(x)
        #print("9: " + str(feature.shape))
        z = self.FC(feature.view(feature.shape[0], -1))
        #print("10: " + str(z.shape))
        return encoder_out, feature, z, cam_logit, heatmap


class CREncoder2(nn.Module):
    def __init__(self):
        # assert(n_blocks >= 0)
        super(CREncoder2, self).__init__()
        self.input_nc = 3
        self.output_nc = 256
        # self.ngf = 64
        self.n_blocks = 6
        self.img_size = 256
        self.light = False

        # in:256x256 out:256x256
        self.DownBlock1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(self.input_nc, 64, kernel_size=3, stride=1, padding=0, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(True))
        # in:256x256 out:128x128
        self.DownBlock2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(True))
        # in:128x128 out:64x64
        self.DownBlock3 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True))

        self.content_ResBlock = nn.Sequential(
            ResnetBlock(256, use_bias=False))
            # nn.UpsamplingBilinear2d(scale_factor=2),
            # nn.ReflectionPad2d(1),
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0, bias=False),
            # nn.InstanceNorm2d(256),
            # nn.ReLU(True))

        # in:64x64 out:32x32
        self.DownBlock4 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True))

        # in:32x32 out:16x16
        self.DownBlock5 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0, groups=256, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True))

        # in:16x16 out:8x8
        self.DownBlock6 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0, groups=256, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(True))

        # Down-Sampling Bottleneck
        self.ResBlock = nn.Sequential(
            ResnetBlock(256, use_bias=False),
            ResnetBlock(256, use_bias=False),
            ResnetBlock(256, use_bias=False))
        
        # Class Activation Map
        self.gap_fc = nn.Linear(256, 1, bias=False)
        self.gmp_fc = nn.Linear(256, 1, bias=False)
        self.conv1x1 = nn.Conv2d(256 * 2, 256, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(True)

        # z generator
        self.FC = nn.Sequential(
            nn.Linear(256*8*8, 256, bias=False),
            nn.LeakyReLU(True),
            nn.Linear(256, 256, bias=False))

    def forward(self, input):
        # downsample
        x = self.DownBlock1(input)
        #print("1: " + str(x.shape))
        x = self.DownBlock2(x)
        # print("2: " + str(x.shape))
        encoder_out_before = self.DownBlock3(x)

        x = self.DownBlock4(encoder_out_before)
        # print("4: " + str(encoder_out_begore.shape))
        x = self.DownBlock5(x)
        # print("5: " + str(x.shape))
        x = self.DownBlock6(x)
        # print("6: " + str(x.shape))

        # CAM
        gap = torch.nn.functional.adaptive_avg_pool2d(encoder_out_before, 1)
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1))
        gap_weight = list(self.gap_fc.parameters())[0]
        gap = encoder_out_before * gap_weight.unsqueeze(2).unsqueeze(3)

        gmp = torch.nn.functional.adaptive_max_pool2d(encoder_out_before, 1)
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1))
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp = encoder_out_before * gmp_weight.unsqueeze(2).unsqueeze(3)

        cam_logit = torch.cat([gap_logit, gmp_logit], 1)
        encoder_out_before = torch.cat([gap, gmp], 1)
        encoder_out_before = self.relu(self.conv1x1(encoder_out_before))
        encoder_out = self.content_ResBlock(encoder_out_before)
        
        heatmap = torch.sum(encoder_out, dim=1, keepdim=True)
        #print("8: " + str(heatmap.shape))
        # print("7encoder_out: " + str(encoder_out.shape))

        feature = self.ResBlock(x)
        #print("9: " + str(feature.shape))
        z = self.FC(feature.view(feature.shape[0], -1))
        #print("10: " + str(z.shape))
        return encoder_out, feature, z, cam_logit, heatmap


class UpsampleBlock(nn.Module):
    def __init__(self, inc, outc, k, s, group):
        super(UpsampleBlock, self).__init__()
        up = []
        up += [nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(inc, outc, kernel_size=k, stride=s, padding=0, bias=False, groups = group)]
        
        self.up = nn.Sequential(*up)

        self.fc = nn.Linear(inc, outc, bias=False)
        self.relu1 = nn.ReLU(True)
        self.adalin = AdaLIN(outc)
        self.relu2 = nn.ReLU(True)

    def forward(self, x, z):
        #print("11111111k:::::::::::::::::: " + str(x.shape))
        x = self.up(x)
        z = self.relu1(self.fc(z))
        x = self.adalin(x, z)
        x = self.relu2(x)
        return x, z

class UpsampleBlock3(nn.Module):
    def __init__(self, inc, outc, k, s, group):
        super(UpsampleBlock3, self).__init__()

        fan_in = inc * 3 ** 2
        self.scale = 1 / math.sqrt(fan_in)

        up = []
        up += [nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(inc, outc, kernel_size=k, stride=s, padding=0, bias=False, groups = group)]
        
        self.up = nn.Sequential(*up)

        self.fc = nn.Linear(512, outc, bias=True)
        self.relu1 = nn.LeakyReLU(0.2, True)
        self.adalin = AdaLIN(outc)
        self.relu2 = nn.LeakyReLU(0.2, True)

        self.rgb_up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.conv2rgb = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(outc, 3, kernel_size=k, stride=s, padding=0, bias=True, groups = 1),
            # nn.LeakyReLU(0.2, True)
        )

        self.rgb_fc = nn.Linear(512, 3, bias=True)
        self.rgb_relu = nn.LeakyReLU(0.2, True)
        self.rgb_adalin = AdaLIN(3)

    def forward(self, x, z, rgb):
        x = self.up(x)*self.scale
        # print("11111111k:::::::::::::::::: " + str(x.shape))
        z1 = self.relu1(self.fc(z))
        # print("11111112k:::::::::::::::::: " + str(z.shape))
        x = self.adalin(x, z1)
        x = self.relu2(x)

        z_rgb = self.rgb_relu(self.rgb_fc(z))
        ttt = self.rgb_adalin(self.conv2rgb(x), z_rgb)

        rgb = self.rgb_up(rgb) + ttt*self.scale
        return x, rgb


class UpsampleBlock2(nn.Module):
    def __init__(self, inc, outc, k, s, group):
        super(UpsampleBlock2, self).__init__()
        up = []
        up += [nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(inc, outc, kernel_size=k, stride=s, padding=0, bias=False, groups = group)]
        
        self.up = nn.Sequential(*up)
        self.lin = LayerInstanceNorm(outc)

    def forward(self, x, z):
        #print("11111111k:::::::::::::::::: " + str(x.shape))
        x = self.up(x)
        x = self.lin(x)
        return x

class CRDecoder(nn.Module):
    def __init__(self):
        # assert(n_blocks >= 0)
        super(CRDecoder, self).__init__()
        self.input_nc = 256
        self.output_nc = 3
        # self.ngf = 64
        self.n_blocks = 3
        self.img_size = 256
        self.light = False

        # MLP
        self.mlp = MLP(256, 256, 8)
        adain_resblock = []
        for i in range(self.n_blocks):
            adain_resblock.append(ResBlockByAdaLIN(256))
        self.adain_resblocks = nn.ModuleList(adain_resblock)

        # in:64x64 out:128x128
        self.upsample1 = UpsampleBlock2(256, 64, k = 3, s = 1, group=4)

        # in:128x128 out:256x256
        self.upsample2 = UpsampleBlock2(64, 32, k = 3, s = 1, group=2)

        # final
        final = [nn.ReflectionPad2d(3),
            nn.Conv2d(32, self.output_nc, kernel_size=7, stride=1, padding=0, bias=False, groups=1),
            nn.Tanh()]
        self.final = nn.Sequential(*final)

    def forward(self, x, z):
        z = self.mlp(z)  # b, 256
        #print("11: " + str(z.shape))
        #print("12: " + str(x.shape))
        for i in range(self.n_blocks):
            x = self.adain_resblocks[i](x, z)
        #print("12k:::::::::::::::::: " + str(x.shape))
        x = self.upsample1(x, z)
        x = self.upsample2(x, z)
        out = self.final(x)
        return out

class CRDecoder_rose(nn.Module):
    def __init__(self):
        # assert(n_blocks >= 0)
        super(CRDecoder_rose, self).__init__()
        self.input_nc = 512
        self.output_nc = 3
        # self.ngf = 64
        self.n_blocks = 3
        self.img_size = 256
        self.light = False

        # MLP
        self.mlp = MLP(8, 512, 8)
        adain_resblock = []
        for i in range(self.n_blocks):
            adain_resblock.append(ResBlockByAdaLIN(512))
        self.adain_resblocks = nn.ModuleList(adain_resblock)

        # in:16x16 out:32x32
        self.upsample1 = UpsampleBlock(512, 256, k = 3, s = 1, group=4)

        # in:32x32 out:64x64
        self.upsample2 = UpsampleBlock(256, 128, k = 3, s = 1, group=4)

        # in:64x64 out:128x128
        self.upsample3 = UpsampleBlock(128, 64, k = 3, s = 1, group=4)

        # in:128x128 out:256x256
        self.upsample4 = UpsampleBlock(64, 32, k = 3, s = 1, group=2)

        # final
        final = [nn.ReflectionPad2d(3),
            nn.Conv2d(32, self.output_nc, kernel_size=7, stride=1, padding=0, bias=False, groups=1),
            nn.Tanh()]
        self.final = nn.Sequential(*final)

        self.initialize_module(self)

    def initialize_module(self, module):
        for m in module.modules():
            # print(":::::::::::::::::::")
            # print(m)
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, x, z):
        z = self.mlp(z)  # b, 256
        #print("11: " + str(z.shape))
        #print("12: " + str(x.shape))
        for i in range(self.n_blocks):
            x = self.adain_resblocks[i](x, z)
        #print("12k:::::::::::::::::: " + str(x.shape))
        x, z = self.upsample1(x, z)
        x, z = self.upsample2(x, z)
        x, z = self.upsample3(x, z)
        x, z = self.upsample4(x, z)
        out = self.final(x)
        return out

class CRDecoder_rose3(nn.Module):
    def __init__(self):
        # assert(n_blocks >= 0)
        super(CRDecoder_rose3, self).__init__()
        self.input_nc = 512
        self.output_nc = 3
        # self.ngf = 64
        self.n_blocks = 1
        self.img_size = 256
        self.light = False

        fan_in = self.input_nc * 3 ** 2
        self.scale = 1 / math.sqrt(fan_in)

        # MLP
        self.mlp = MLP(8, 512, 8)
        adain_resblock = []
        for i in range(self.n_blocks):
            adain_resblock.append(ResBlockByAdaLIN(512))
        self.adain_resblocks = nn.ModuleList(adain_resblock)

        self.conv2rgb = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 3, kernel_size=3, stride=1, padding=0, bias=True, groups = 1),
            # nn.LeakyReLU(0.2, True)
        )
        self.rgb_fc = nn.Linear(512, 3, bias=True)
        self.rgb_relu = nn.LeakyReLU(0.2, True)
        self.rgb_adalin = AdaLIN(3)

        # in:16x16 out:32x32
        self.upsample1 = UpsampleBlock3(512, 256, k = 3, s = 1, group=4)

        # in:32x32 out:64x64
        self.upsample2 = UpsampleBlock3(256, 128, k = 3, s = 1, group=4)

        # in:64x64 out:128x128
        self.upsample3 = UpsampleBlock3(128, 64, k = 3, s = 1, group=4)

        # in:128x128 out:256x256
        self.upsample4 = UpsampleBlock3(64, 32, k = 3, s = 1, group=2)

        # final
        final = [nn.ReflectionPad2d(3),
            nn.Conv2d(32, self.output_nc, kernel_size=7, stride=1, padding=0, bias=False, groups=1),
            nn.Tanh()]
        self.final = nn.Sequential(*final)

        self.initialize_module(self)

    def initialize_module(self, module):
        for m in module.modules():
            # print(":::::::::::::::::::")
            # print(m)
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, z):
        z = self.mlp(z)  # b, 256
        # print("11: " + str(z))
        #print("12: " + str(x.shape))
        for i in range(self.n_blocks):
            x = self.adain_resblocks[i](x, z)*self.scale

        # print (x.shape)
        rgb_z = self.rgb_relu(self.rgb_fc(z))
        rgb = self.rgb_adalin(self.conv2rgb(x), rgb_z)*self.scale
        # rgb = self.conv2rgb(x)*self.scale
        # print (rgb)
        
        x, rgb = self.upsample1(x, z, rgb)
        x, rgb = self.upsample2(x, z, rgb)
        x, rgb = self.upsample3(x, z, rgb)
        x, rgb = self.upsample4(x, z, rgb)
        # out = self.final(x)
        return rgb

class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetAdaILNBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x


class adaILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)

        return out


class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)

        return out


class RhoClipper(object):

    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):

        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w

from torch.nn import functional as F
channels = {
    4: 512,
    8: 512,
    16: 512,
    32: 512,
    64: 256,
    128: 128,
    256: 64,
    512: 32,
    1024: 16,
}
channels2 = {
    16: 512,
    32: 128,
    64: 64,
    128: 32,
    256: 16,
}
class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()

        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    rest_dim = [1] * (input.ndim - bias.ndim - 1)
    if input.ndim == 3:
        return (
            F.leaky_relu(
                input + bias.view(1, *rest_dim, bias.shape[0]), negative_slope=negative_slope
            )
            * scale
        )
    else:
        return (
            F.leaky_relu(
                input + bias.view(1, bias.shape[0], *rest_dim), negative_slope=negative_slope
            )
            * scale
        )

def upfirdn2d_native(
    input, kernel, up_x, up_y, down_x, down_y, pad_x0, pad_x1, pad_y0, pad_y1
):
    _, channel, in_h, in_w = input.shape
    input = input.reshape(-1, in_h, in_w, 1)

    _, in_h, in_w, minor = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input.view(-1, in_h, 1, in_w, 1, minor)
    out = F.pad(out, [0, 0, 0, up_x - 1, 0, 0, 0, up_y - 1])
    out = out.view(-1, in_h * up_y, in_w * up_x, minor)

    out = F.pad(
        out, [0, 0, max(pad_x0, 0), max(pad_x1, 0), max(pad_y0, 0), max(pad_y1, 0)]
    )
    out = out[
        :,
        max(-pad_y0, 0) : out.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : out.shape[2] - max(-pad_x1, 0),
        :,
    ]

    out = out.permute(0, 3, 1, 2)
    out = out.reshape(
        [-1, 1, in_h * up_y + pad_y0 + pad_y1, in_w * up_x + pad_x0 + pad_x1]
    )
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    out = F.conv2d(out, w)
    out = out.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )
    out = out.permute(0, 2, 3, 1)
    out = out[:, ::down_y, ::down_x, :]

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    return out.view(-1, channel, out_h, out_w)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k

def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    out = upfirdn2d_native(
        input, kernel, up, up, down, down, pad[0], pad[1], pad[0], pad[1]
    )

    return out

class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out

class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out

class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        bias = self.bias*self.lr_mul if self.bias is not None else None
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, bias)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=bias
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )

# class ModulatedConv2d(nn.Module):
#     def __init__(
#         self,
#         in_channel,
#         out_channel,
#         kernel_size,
#         style_dim,
#         use_style=True,
#         demodulate=True,
#         upsample=False,
#         downsample=False,
#         blur_kernel=[1, 3, 3, 1],
#     ):
#         super().__init__()

#         self.eps = 1e-8
#         self.kernel_size = kernel_size
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#         self.upsample = upsample
#         self.downsample = downsample
#         self.use_style = use_style

#         if upsample:
#             factor = 2
#             p = (len(blur_kernel) - factor) - (kernel_size - 1)
#             pad0 = (p + 1) // 2 + factor - 1
#             pad1 = p // 2 + 1

#             self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

#         if downsample:
#             factor = 2
#             p = (len(blur_kernel) - factor) + (kernel_size - 1)
#             pad0 = (p + 1) // 2
#             pad1 = p // 2

#             self.blur = Blur(blur_kernel, pad=(pad0, pad1))

#         fan_in = in_channel * kernel_size ** 2
#         self.scale = 1 / math.sqrt(fan_in)
#         self.padding = kernel_size // 2

#         self.weight = nn.Parameter(
#             torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
#         )

#         if use_style:
#             self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
#         else:
#             self.modulation = nn.Parameter(torch.Tensor(1, 1, in_channel, 1, 1).fill_(1))

#         self.demodulate = demodulate

#     def __repr__(self):
#         return (
#             f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
#             f'upsample={self.upsample}, downsample={self.downsample})'
#         )

#     def forward(self, input, style):
#         batch, in_channel, height, width = input.shape

#         if self.use_style:
#             style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
#             weight = self.scale * self.weight * style
#         else:
#             weight = self.scale * self.weight.expand(batch,-1,-1,-1,-1) * self.modulation

#         if self.demodulate:
#             demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
#             weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

#         weight = weight.view(
#             batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
#         )

#         if self.upsample:
#             input = input.view(1, batch * in_channel, height, width)
#             weight = weight.view(
#                 batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
#             )
#             weight = weight.transpose(1, 2).reshape(
#                 batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
#             )
#             out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
#             _, _, height, width = out.shape
#             out = out.view(batch, self.out_channel, height, width)
#             out = self.blur(out)

#         elif self.downsample:
#             input = self.blur(input)
#             _, _, height, width = input.shape
#             input = input.view(1, batch * in_channel, height, width)
#             out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
#             _, _, height, width = out.shape
#             out = out.view(batch, self.out_channel, height, width)

#         else:
#             input = input.view(1, batch * in_channel, height, width)
#             out = F.conv2d(input, weight, padding=self.padding, groups=batch)
#             _, _, height, width = out.shape
#             out = out.view(batch, self.out_channel, height, width)

#         return out

class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        use_style=True,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        cgroup = 1,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        self.use_style = use_style
        self.cgroup = cgroup

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)
            # self.blur = nn.Sequential(
            #     nn.ConstantPad2d((0, -1, 0,-1), 0),
            #     nn.Upsample(scale_factor=1, mode="bilinear"))

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        if upsample:
            self.weight = nn.Parameter(
                torch.randn(1, out_channel//2, in_channel, kernel_size, kernel_size)
            )
        else:
            self.weight = nn.Parameter(
                torch.randn(1, out_channel, in_channel//self.cgroup, kernel_size, kernel_size)
            )

        # self.wl = nn.ReLU6(nn.Tanh())
        # self.convs = nn.Conv2d(1 * in_channel, self.out_channel, self.kernel_size, stride=1,groups=cgroup,padding=self.padding)
        # self.convs_d = nn.Conv2d(1 * in_channel, self.out_channel, self.kernel_size, stride=2,groups=1,padding=0)
        # self.convs_t = nn.ConvTranspose2d(1 * in_channel, self.out_channel, self.kernel_size, stride=2,groups=2,padding=0)

        if use_style:
            self.modulation = EqualLinear(style_dim, in_channel//self.cgroup, bias_init=1)
            # self.modulation = nn.Sequential(
            #     nn.LeakyReLU(negative_slope=0.2),
            #     nn.Linear(style_dim, in_channel//self.cgroup))
        else:
            self.modulation = nn.Parameter(torch.Tensor(1, 1, in_channel, 1, 1).fill_(1))

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        if self.use_style:
            if self.upsample:
                style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
            else:
                style = self.modulation(style).view(batch, 1, in_channel//self.cgroup, 1, 1)
            weight = self.scale * self.weight * style
        else:
            weight = self.scale * self.weight.expand(batch,-1,-1,-1,-1) * self.modulation

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            if self.upsample:
                weight = weight * demod.view(batch, self.out_channel//2, 1, 1, 1)
            else:
                weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        if self.upsample:
            weight = weight.view(
                batch * self.out_channel//2, in_channel, self.kernel_size, self.kernel_size
            )
        else:
            weight = weight.view(
                batch * self.out_channel, in_channel//self.cgroup, self.kernel_size, self.kernel_size
            )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel//2, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel//2, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch*2)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch*self.cgroup)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out

class ModulatedConv2d2(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        use_style=True,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        cgroup = 1,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        self.use_style = use_style
        self.cgroup = cgroup

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            # self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)
            self.blur = nn.Sequential(
                nn.ConstantPad2d((0, -1, 0,-1), 0),
                nn.Upsample(scale_factor=1, mode="bilinear"))

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        if upsample:
            self.weight = nn.Parameter(
                torch.randn(1, out_channel//2, in_channel, kernel_size, kernel_size)
            )
        else:
            self.weight = nn.Parameter(
                torch.randn(1, out_channel, in_channel//self.cgroup, kernel_size, kernel_size)
            )

        self.wl = nn.ReLU6(nn.Tanh())
        self.convs = nn.Conv2d(1 * in_channel, self.out_channel, self.kernel_size, stride=1,groups=cgroup,padding=self.padding)
        self.convs_d = nn.Conv2d(1 * in_channel, self.out_channel, self.kernel_size, stride=2,groups=1,padding=0)
        self.convs_t = nn.ConvTranspose2d(1 * in_channel, self.out_channel, self.kernel_size, stride=2,groups=2,padding=0)

        if use_style:
            # self.modulation = EqualLinear(style_dim, in_channel//self.cgroup, bias_init=1)
            self.modulation = nn.Sequential(
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(style_dim, in_channel//self.cgroup))
        else:
            self.modulation = nn.Parameter(torch.Tensor(1, 1, in_channel, 1, 1).fill_(1))

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        if self.use_style:
            if self.upsample:
                style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
            else:
                style = self.modulation(style).view(batch, 1, in_channel//self.cgroup, 1, 1)
            weight = self.scale * self.weight * style
            weight = self.wl(weight)
        else:
            weight = self.scale * self.weight.expand(batch,-1,-1,-1,-1) * self.modulation

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            if self.upsample:
                weight = weight * demod.view(batch, self.out_channel//2, 1, 1, 1)
            else:
                weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        if self.upsample:
            weight = weight.view(
                batch * self.out_channel//2, in_channel, self.kernel_size, self.kernel_size
            )
        else:
            weight = weight.view(
                batch * self.out_channel, in_channel//self.cgroup, self.kernel_size, self.kernel_size
            )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel//2, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel//2, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch*2)
            # out = self.convs_t(input)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            # print(out.shape)
            out = self.blur(out)
            # print(out.shape)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            # out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            out = self.convs_d(input)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            # out = F.conv2d(input, weight, padding=self.padding, groups=batch*self.cgroup)
            out = self.convs(input)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out

class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        use_style=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
        activation = True,
        group = 1,
    ):
        super().__init__()
        self.use_style = use_style
        self.activation = activation

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            use_style=use_style,
            upsample=upsample,
            downsample=downsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
            cgroup=group,
        )

        #if use_style:
        #    self.noise = NoiseInjection()
        #else:
        #    self.noise = None
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style=None, noise=None):
        out = self.conv(input, style)
        #if self.use_style:
        #    out = self.noise(out, noise=noise)
        # out = out + self.bias
        if self.activation:
            out = self.activate(out)

        return out

class StyledConv2(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        use_style=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()
        self.use_style = use_style

        self.conv = ModulatedConv2d2(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            use_style=use_style,
            upsample=upsample,
            downsample=downsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        # self.conv = nn.Conv2d(in_channel, out_channel,3,1,0, groups=2)

        #if use_style:
        #    self.noise = NoiseInjection()
        #else:
        #    self.noise = None
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)

        # self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style=None, noise=None):
        out = self.conv(input, style)
        # out = self.conv(input)
        #if self.use_style:
        #    out = self.noise(out, noise=noise)
        # out = out + self.bias
        # out = self.activate(out)
        return out

class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 4, 1, style_dim, demodulate=False, cgroup=1)
        self.bias = nn.Parameter(torch.zeros(1, 4, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out

class ToRGB2(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            # self.upsample = Upsample(blur_kernel)
            self.upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear"))

        self.conv = ModulatedConv2d2(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out

class StyledResBlock(nn.Module):
    def __init__(self, in_channel, style_dim, blur_kernel=[1, 3, 3, 1], demodulate=True):
        super().__init__()
        self.conv1 = StyledConv(in_channel, in_channel*2, 1, style_dim, upsample=False, blur_kernel=blur_kernel, demodulate=demodulate)
        self.conv2 = StyledConv(in_channel*2, in_channel*2, 3, style_dim, upsample=False, blur_kernel=blur_kernel, demodulate=demodulate, group=in_channel)
        self.conv3 = StyledConv(in_channel*2, in_channel, 1, style_dim, upsample=False, blur_kernel=blur_kernel, demodulate=demodulate, activation=False)

        # self.conv1 = StyledConv(in_channel, in_channel//2, 3, style_dim, upsample=False, blur_kernel=blur_kernel, demodulate=demodulate)
        # self.conv2 = StyledConv(in_channel//2, in_channel, 3, style_dim, upsample=False, blur_kernel=blur_kernel, demodulate=demodulate)

    def forward(self, input, style):
        out = self.conv1(input, style)
        out = self.conv2(out, style)
        out = self.conv3(out, style)
        out = (out + input) / math.sqrt(2)

        return out

class StyledRes(nn.Module):
    def __init__(self, in_channel, style_dim, blur_kernel=[1, 3, 3, 1], demodulate=True, largec=True):
        super().__init__()
        k = 1
        if largec:
            k = 2
        self.conv1 = StyledConv(in_channel, in_channel*k, 1, style_dim, upsample=False, blur_kernel=blur_kernel, demodulate=demodulate)
        self.conv2 = StyledConv(in_channel*k, in_channel*k, 3, style_dim, upsample=False, blur_kernel=blur_kernel, demodulate=demodulate, group=in_channel)
        self.conv3 = StyledConv(in_channel*k, in_channel, 1, style_dim, upsample=False, blur_kernel=blur_kernel, demodulate=demodulate, activation=False)

        # self.conv1 = StyledConv(in_channel, in_channel//2, 3, style_dim, upsample=False, blur_kernel=blur_kernel, demodulate=demodulate)
        # self.conv2 = StyledConv(in_channel//2, in_channel, 3, style_dim, upsample=False, blur_kernel=blur_kernel, demodulate=demodulate)

    def forward(self, input, style):
        out = self.conv1(input, style)
        out = self.conv2(out, style)
        out = self.conv3(out, style)
        # out = (out + input) / math.sqrt(2)
        return out


#1.4G flops
class Decoder_kkk(nn.Module):
    def __init__(
        self
    ):
        # assert(n_blocks >= 0)
        super(Decoder_kkk, self).__init__()
        self.input_nc = 512
        self.output_nc = 3
        self.size = 256
        self.num_down = 4
        self.latent_dim = 8
        self.n_mlp = 5
        channel_multiplier=1
        blur_kernel=[1, 3, 3, 1]
        lr_mlp=0.01

        # self.use_mapping=True
        self.log_size = int(math.log(self.size, 2)) #7
        in_log_size = self.log_size - self.num_down #7-2 or 7-3
        in_size = 2 ** in_log_size

        style_dim = 512
        in_channel = 512
        # MLP
        self.mapping = MLP(self.latent_dim, style_dim, self.n_mlp)
        
        self.adain_bottleneck = nn.ModuleList()
        for i in range(2):
            self.adain_bottleneck.append(StyledResBlock(in_channel, style_dim))

        self.conv1 = StyledConv(in_channel, 512, 1, style_dim, upsample=False, blur_kernel=blur_kernel)
        self.conv1_s = StyledConv(512, 512, 3, style_dim, upsample=False, blur_kernel=blur_kernel, group=512)
        self.conv1_ss = StyledConv(512, 256, 1, style_dim, upsample=False, blur_kernel=blur_kernel, activation=False)

        self.to_rgb1 = ToRGBcr(256, style_dim, upsample=False)

        self.to_alpha = nn.Sigmoid()

        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        in_channel = 256
        for i in range(in_log_size+1, self.log_size + 1):
            out_channel = channels2[2 ** i]
            # print(":::::kk:::", out_channel)

            self.convs.append(
                StyledConv2dUpcr(in_channel, out_channel, style_dim)
                # StyledConv(
                #     in_channel,
                #     out_channel,
                #     3,
                #     style_dim,
                #     upsample=True,
                #     blur_kernel=blur_kernel,
                # )
            )

            # self.convs.append(
            #     StyledConv(
            #         out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
            #     )
            # )
            self.convs.append(
                StyledConv2dcr(out_channel, out_channel, style_dim, 3)
                # StyledRes(out_channel, style_dim,largec=False)
            )

            self.to_rgbs.append(ToRGBcr(out_channel, style_dim))

            in_channel = out_channel

    def forward(self, input, styles):
        styles = self.mapping(styles)
        #styles = styles.repeat(1, n_latent).view(styles.size(0), n_latent, -1)
        out = input
        i = 0
        for conv in self.adain_bottleneck:
            out = conv(out, styles)
            i += 1

        out = self.conv1(out, styles, noise=None)
        out = self.conv1_s(out, styles, noise=None)
        out = self.conv1_ss(out, styles, noise=None)
        skip = self.to_rgb1(out, styles)
        i += 2

        for conv1, conv2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], self.to_rgbs
        ):
            out = conv1(out, styles, noise=None)
            out = conv2(out, styles)
            skip = to_rgb(out, styles, skip)

            i += 3

        image = skip[:,0:3,:,:]
        alpha = self.to_alpha(skip[:,3:4,:,:])

        return image, alpha


from modulated_conv2d import *
# from modulated_conv2d import ModulatedDWConv2dcr
class Decoder_kkk2(nn.Module):
    def __init__(
        self
    ):
        # assert(n_blocks >= 0)
        super(Decoder_kkk2, self).__init__()
        self.input_nc = 512
        self.output_nc = 3
        self.size = 256
        self.num_down = 4
        self.latent_dim = 8
        self.n_mlp = 5
        channel_multiplier=1
        blur_kernel=[1, 3, 3, 1]
        lr_mlp=0.01

        # self.use_mapping=True
        self.log_size = int(math.log(self.size, 2)) #7
        in_log_size = self.log_size - self.num_down #7-2 or 7-3
        in_size = 2 ** in_log_size

        style_dim = 512
        in_channel = 512
        # MLP
        self.mapping = MLP(self.latent_dim, style_dim, self.n_mlp)
        
        self.adain_bottleneck = nn.ModuleList()
        for i in range(2):
            self.adain_bottleneck.append(StyledResBlockcr(in_channel, style_dim))

        # self.conv1 = StyledConv2dcr(in_channel, 512, style_dim, 3)
        self.conv1 = StyledConv2dcr(in_channel, 256, style_dim, 3)

        # self.conv1 = StyledConv(in_channel, 512, 1, style_dim, upsample=False, blur_kernel=blur_kernel)
        # self.conv1_s = StyledConv(512, 512, 3, style_dim, upsample=False, blur_kernel=blur_kernel, group=512)
        # self.conv1_ss = StyledConv(512, 256, 1, style_dim, upsample=False, blur_kernel=blur_kernel, activation=False)

        self.to_rgb1 = ToRGBcr(256, style_dim, upsample=False)

        self.to_alpha = nn.Sigmoid()

        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        in_channel = 256
        for i in range(in_log_size+1, self.log_size + 1):
            out_channel = channels2[2 ** i]
            # print(":::::kk:::", out_channel)

            self.convs.append(
                StyledConv2dUpcr(in_channel, out_channel, style_dim)
            )

            self.convs.append(
                StyledConv2dcr(out_channel, out_channel, style_dim, 3)
            )

            self.to_rgbs.append(ToRGBcr(out_channel, style_dim))

            in_channel = out_channel

    def forward(self, input, styles):
        styles = self.mapping(styles)
        #styles = styles.repeat(1, n_latent).view(styles.size(0), n_latent, -1)
        out = input
        i = 0
        for conv in self.adain_bottleneck:
            out = conv(out, styles)
            i += 1

        out = self.conv1(out, styles, noise=None)
        skip = self.to_rgb1(out, styles)
        i += 2

        for conv1, conv2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], self.to_rgbs
        ):
            out = conv1(out, styles, noise=None)
            out = conv2(out, styles)
            skip = to_rgb(out, styles, skip)

            i += 3

        image = skip[:,0:3,:,:]
        alpha = self.to_alpha(skip[:,3:4,:,:])

        return image, alpha

class Decoder_kkk512(nn.Module):
    def __init__(
        self
    ):
        # assert(n_blocks >= 0)
        super(Decoder_kkk512, self).__init__()
        self.input_nc = 512
        self.output_nc = 3
        self.size = 256
        self.num_down = 4
        self.latent_dim = 8
        self.n_mlp = 5
        channel_multiplier=1
        blur_kernel=[1, 3, 3, 1]
        lr_mlp=0.01

        # self.use_mapping=True
        self.log_size = int(math.log(self.size, 2)) #7
        in_log_size = self.log_size - self.num_down #7-2 or 7-3
        in_size = 2 ** in_log_size

        style_dim = 512
        in_channel = 512
        # MLP
        self.mapping = MLP(self.latent_dim, style_dim, self.n_mlp)
        
        self.adain_bottleneck = nn.ModuleList()
        for i in range(2):
            self.adain_bottleneck.append(StyledResBlockcr(in_channel, style_dim))

        # self.conv1 = StyledConv2dcr(in_channel, 512, style_dim, 3)
        self.conv1 = StyledConv2dcr(in_channel, 256, style_dim, 3)

        # self.conv1 = StyledConv(in_channel, 512, 1, style_dim, upsample=False, blur_kernel=blur_kernel)
        # self.conv1_s = StyledConv(512, 512, 3, style_dim, upsample=False, blur_kernel=blur_kernel, group=512)
        # self.conv1_ss = StyledConv(512, 256, 1, style_dim, upsample=False, blur_kernel=blur_kernel, activation=False)

        self.to_rgb1 = ToRGBcr(256, style_dim, upsample=False)

        self.to_alpha = nn.Sigmoid()

        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        in_channel = 256
        for i in range(in_log_size+1, self.log_size + 1):
            out_channel = channels2[2 ** i]
            # print(":::::kk:::", out_channel)

            self.convs.append(
                StyledConv2dUpcr(in_channel, out_channel, style_dim)
            )

            self.convs.append(
                StyledConv2dcr(out_channel, out_channel, style_dim, 3)
            )

            self.to_rgbs.append(ToRGBcr(out_channel, style_dim))

            in_channel = out_channel

        self.ups2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv512 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 9, kernel_size=3, stride=1, padding=0, bias=False),
            # nn.InstanceNorm2d(9),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(9, 3, kernel_size=3, stride=1, padding=0, bias=False),
            # nn.InstanceNorm2d(3),
        )

    def forward(self, input, styles):
        styles = self.mapping(styles)
        #styles = styles.repeat(1, n_latent).view(styles.size(0), n_latent, -1)
        out = input
        i = 0
        for conv in self.adain_bottleneck:
            out = conv(out, styles)
            i += 1

        out = self.conv1(out, styles, noise=None)
        skip = self.to_rgb1(out, styles)
        i += 2

        for conv1, conv2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], self.to_rgbs
        ):
            out = conv1(out, styles, noise=None)
            out = conv2(out, styles)
            skip = to_rgb(out, styles, skip)

            i += 3

        image256 = skip[:,0:3,:,:]
        alpha = self.to_alpha(skip[:,3:4,:,:])

        image512 = self.ups2(image256)

        delta = 0.01*self.conv512(image512)

        image512 = image512 + delta

        return image256, alpha, image512


from thop import profile
from matplotlib import pyplot as plt 
from skimage import io
from skimage import transform
import numpy as np
# from torchstat import stat
# from ptflops import get_model_complexity_info


def count_your_model(model):
    sum = 0
    for m in model.modules():
        print(m)
        if isinstance(m, (nn.Conv2d)):
            sum = (m.weight.shape[2]*m.weight.shape[3]*m.in_channels*m.out_channels+m.out_channels)*256*256
            print(m.weight.shape[0], m.weight.shape[1],m.weight.shape[2],m.weight.shape[3])

        elif isinstance(m, ModulatedConv2d2):
            sum = (m.weight.shape[2]*m.weight.shape[3]*m.in_channel*m.out_channel+m.out_channel)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    print(sum)
    return 10000
    #return y.size(2) * y.size(3) * y.size(1) * self.conv.weight.size(2) * self.conv.weight.size(3) / 1


def test():
    torch.set_printoptions(precision=4, sci_mode=False)
    # net = CRGenerator_small(3, 3, 32, 3, 256, False)

    # net = CRDecoder_rose3()

    net = Decoder_kkk2()
    
    style = torch.randn(1, 8)
    style2 = torch.randn(1, 8)
    style3 = torch.randn(2, 8)

    # z_bb = torch.randn(1, 256)*100
    # noo = nn.LayerNorm(256)
    # z_bb = noo(z_bb)
    # z_kk = 0

    x1 = torch.randn(1,3,256,256)
    x2 = torch.randn(1,512,16,16)
    x3 = torch.randn(2,512,16,16)


    img = io.imread("/Users/cr/git/face/Morph-UGATIT/datasets/trainA/female_4493.jpg")

    img = transform.resize(img, (16, 16))
    img_tensor = torch.from_numpy(img.transpose((2,0,1)))
    img_tensor = torch.reshape(img_tensor, (1, 3, 16, 16))[:,0:2,:,:]
    # x2 = img_tensor.repeat((1, 256, 1, 1)).float()
    # print (x2.shape)
    # print(x2)


    out, alpha = net(x2, style)
    out2, alpha = net(x2, style2)
    # print(alpha)
    # stat(net, (3, 256, 256) )
    #flops,params = get_model_complexity_info(net,(1,3,256,256),as_strings=True,print_per_layer_stat=True)
    # print("________________________\n" + str(net))

    million = 100 * 10000
    flops256, _ = profile(net, (x2,style,))
    print("decoder flops", flops256/million)
    total = sum([param.nelement() for param in net.parameters()])
    print("params:::::::::", total/million)


    # net2 = MLP(8, 512, 8)
    xx = torch.randn(1,4,128,128)
    styles = torch.randn(1, 512)
    net2 = StyledConv2(4, 6, 3, 512, upsample=True)
    # net2 = torch.nn.DataParallel(net2)
    flops2, _ = profile(net2, (xx, styles, ))
    #count_your_model(net2)
    # print("decoder flops:::::::::", flops2/million)
    

    # print (out.shape)
    out = torch.reshape(out, (3, 256, 256))
    out = out.detach().numpy().transpose((1, 2, 0))
    out = out - np.min(out)/(np.max(out)-np.min(out))

    out2 = torch.reshape(out2, (3, 256, 256))
    out2 = out2.detach().numpy().transpose((1, 2, 0))
    out2 = out2 - np.min(out2)/(np.max(out2)-np.min(out2))
    plt.figure("haha")
    plt.subplot(1,2,1), plt.title('a')
    plt.imshow(out)
    plt.subplot(1,2,2), plt.title('b')
    plt.imshow(out2)

    plt.show()

    # realA = torch.randn(1,2,4,4)
    # realA_filp=torch.flip(realA,[3])

    # print(realA)
    # print(realA_filp)

    # print(z_bb)
    # print(z.shape)
    # print(cam_logit)


    # print(z_bb)
    

# test()
