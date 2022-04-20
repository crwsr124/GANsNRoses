import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class DWT(torch.nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4

        # return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)
        return x_LL, x_HL, x_LH, x_HH

class DWTLoss(torch.nn.Module):
    def __init__(self):
        super(DWTLoss, self).__init__()
        self.dwt = DWT()
        self.l1 = nn.L1Loss()

    def forward(self, x, x_g):
        x_LL1, x_HL1, x_LH1, x_HH1 = self.dwt(x)
        x_LL2, x_HL2, x_LH2, x_HH2 = self.dwt(x_LL1)
        x_LL3, x_HL3, x_LH3, x_HH3 = self.dwt(x_LL2)

        x_LL1_g, x_HL1_g, x_LH1_g, x_HH1_g = self.dwt(x_g)
        x_LL2_g, x_HL2_g, x_LH2_g, x_HH2_g = self.dwt(x_LL1_g)
        x_LL3_g, x_HL3_g, x_LH3_g, x_HH3_g = self.dwt(x_LL2_g)

        loss = self.l1(x_HL1, x_HL1_g) + self.l1(x_LH1, x_LH1_g) + self.l1(x_HH1, x_HH1_g) +\
            self.l1(x_HL2, x_HL2_g) + self.l1(x_LH2, x_LH2_g) + self.l1(x_HH2, x_HH2_g) +\
            self.l1(x_HL3, x_HL3_g) + self.l1(x_LH3, x_LH3_g) + self.l1(x_HH3, x_HH3_g)
        return loss

class ColorLoss(nn.Module):
    def __init__(self):
        super(ColorLoss, self).__init__()
        self.l1 = nn.L1Loss()
        self.huber = nn.SmoothL1Loss()
        self.rgb_to_yuv_kernel = torch.tensor([
                [0.299, -0.14714119, 0.61497538],
                [0.587, -0.28886916, -0.51496512],
                [0.114, 0.43601035, -0.10001026]
            ]).float().to('cuda')

    def forward(self, image, image_g):
        image = self.rgb_to_yuv(image)
        image_g = self.rgb_to_yuv(image_g)

        # After convert to yuv, both images have channel last

        return (self.l1(image[:, :, :, 0], image_g[:, :, :, 0]) +
                self.huber(image[:, :, :, 1], image_g[:, :, :, 1]) +
                self.huber(image[:, :, :, 2], image_g[:, :, :, 2]))

    def rgb_to_yuv(self, image):
        '''
        https://en.wikipedia.org/wiki/YUV
        output: Image of shape (H, W, C) (channel last)
        '''
        # -1 1 -> 0 1
        image = (image + 1.0) / 2.0

        yuv_img = torch.tensordot(
            image,
            self.rgb_to_yuv_kernel,
            dims=([image.ndim - 3], [0]))
        return yuv_img

class ModulatedConv2d(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            style_dim,
            kernel_size,
            demodulate=True
    ):
        super().__init__()
        # self.module = ModulatedConv2d_origin(channels_in, channels_out, style_dim, kernel_size, demodulate)
        # self.module = WNConv2d_origin(channels_in, channels_out, style_dim, kernel_size, demodulate)
        # self.module = ModulatedConv2d_small(channels_in, channels_out, style_dim, kernel_size, demodulate)
        self.module = WNConv2d_small(channels_in, channels_out, style_dim, kernel_size, demodulate)

    def forward(self, x, style):
        x = self.module(x, style)
        return x

class ModulatedConv2d_origin(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            style_dim,
            kernel_size,
            demodulate=True
    ):
        super().__init__()
        # create conv
        self.weight = nn.Parameter(
            torch.randn(channels_out, channels_in, kernel_size, kernel_size)
        )
        # create modulation network
        self.modulation = nn.Linear(style_dim, channels_in, bias=True)
        self.modulation.bias.data.fill_(1.0)
        # create demodulation parameters
        self.demodulate = demodulate
        if self.demodulate:
            self.register_buffer("style_inv", torch.randn(1, 1, channels_in, 1, 1))
        # some service staff
        self.scale = 1.0 / math.sqrt(channels_in * kernel_size ** 2)
        self.padding = kernel_size // 2

        # self.fconv = nn.Conv2d(channels_in, channels_out, kernel_size,padding=self.padding)

    def forward(self, x, style):
        modulation = self.get_modulation(style)
        x = modulation * x
        x = F.conv2d(x, self.weight, padding=self.padding)
        # x = self.fconv(x)
        if self.demodulate:
            demodulation = self.get_demodulation(style)
            x = demodulation * x
        return x

    def get_modulation(self, style):
        style = self.modulation(style).view(style.size(0), -1, 1, 1)
        modulation = self.scale * style
        return modulation

    def get_demodulation(self, style):
        w = self.weight.unsqueeze(0)
        norm = torch.rsqrt((self.scale * self.style_inv * w).pow(2).sum([2, 3, 4]) + 1e-8)
        demodulation = norm
        return demodulation.view(*demodulation.size(), 1, 1)

class ModulatedConv2d_small(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            style_dim,
            kernel_size,
            demodulate=True
    ):
        super().__init__()
        # create conv
        self.weight_dw = nn.Parameter(
            torch.randn(channels_in, 1, kernel_size, kernel_size)
        )
        self.weight_permute = nn.Parameter(
            torch.randn(channels_out, channels_in, 1, 1)
        )
        # create modulation network
        self.modulation = nn.Linear(style_dim, channels_in, bias=True)
        self.modulation.bias.data.fill_(1.0)
        # create demodulation parameters
        self.demodulate = demodulate
        if self.demodulate:
            self.register_buffer("style_inv", torch.randn(1, 1, channels_in, 1, 1))
        # some service staff
        self.scale = 1.0 / math.sqrt(channels_in * kernel_size ** 2)
        self.padding = kernel_size // 2

    def forward(self, x, style):
        modulation = self.get_modulation(style)
        x = modulation * x
        x = F.conv2d(x, self.weight_dw, padding=self.padding, groups=x.size(1))
        x = F.conv2d(x, self.weight_permute)
        if self.demodulate:
            demodulation = self.get_demodulation(style)
            x = demodulation * x
        return x

    def get_modulation(self, style):
        style = self.modulation(style).view(style.size(0), -1, 1, 1)
        modulation = self.scale * style
        return modulation

    def get_demodulation(self, style):
        w = (self.weight_dw.transpose(0, 1) * self.weight_permute).unsqueeze(0)
        norm = torch.rsqrt((self.scale * self.style_inv * w).pow(2).sum([2, 3, 4]) + 1e-8)
        demodulation = norm
        return demodulation.view(*demodulation.size(), 1, 1)

class MeanOnlyIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(MeanOnlyIN, self).__init__()
        self.eps = eps
        self.mean = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        # out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        out = (input - in_mean)+self.mean
        return out

class AdaMeanOnlyIN(nn.Module):
    def __init__(self, num_features, style_dim, eps=1e-5):
        super(AdaMeanOnlyIN, self).__init__()
        self.eps = eps
        self.modulation = nn.Linear(style_dim, num_features, bias=True)
        self.modulation.bias.data.fill_(1.0)

        self.mean = nn.Parameter(torch.zeros(1, num_features, 1, 1))

        self.scale = 1.0 / math.sqrt(num_features * 3 ** 2)

    def forward(self, input, style):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        # out = (input - in_mean) / torch.sqrt(in_var + self.eps)
        out = (input - in_mean)

        mstyle = self.modulation(style)*self.scale
        mstyle = mstyle.view(mstyle.size(0), -1, 1, 1)
        out = out*mstyle + self.mean

        return out

class WNConv2d_origin(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            style_dim,
            kernel_size,
            demodulate=True
    ):
        super().__init__()
        self.modulation = nn.Linear(style_dim, channels_in, bias=True)
        self.modulation.bias.data.fill_(1.0)
        self.mean_only_in = MeanOnlyIN(channels_out)
        self.scale = 1.0 / math.sqrt(channels_in * kernel_size ** 2)
        self.conv = nn.utils.weight_norm(nn.Conv2d(channels_in, channels_out, kernel_size, padding=kernel_size//2, bias = False))
        # self.conv = nn.Conv2d(channels_in, channels_out, kernel_size, padding=kernel_size//2)
        # self.adain = AdaMeanOnlyIN(channels_out, style_dim)
        self.adain = AdaMeanOnlyIN(channels_in, style_dim)
        self.initialize_module(self)

    def forward(self, x, style):
        # x = self.mean_only_in(x)
        mstyle = self.modulation(style)*self.scale
        mystyle = mstyle.view(mstyle.size(0), -1, 1, 1)
        x = mystyle*x
        x = self.conv(x)
        # x = self.mean_only_in(x)
        return x
    # def forward(self, x, style):
    #     x = self.conv(x)
    #     x = self.adain(x, style)
    #     return x
    # def forward(self, x, style):
    #     x = self.adain(x, style)
    #     x = self.conv(x)
    #     return x

    def initialize_module(self, module):
        for m in module.modules():
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

class WNConv2d_small(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            style_dim,
            kernel_size,
            demodulate=True
    ):
        super().__init__()
        self.adain = AdaMeanOnlyIN(channels_in, style_dim)
        self.mean_only_in = MeanOnlyIN(channels_out)
        self.modulation = nn.Linear(style_dim, channels_in, bias=True)
        nn.init.kaiming_normal_(self.modulation.weight, mode="fan_in", nonlinearity="leaky_relu")
        # nn.init.normal_(self.modulation.weight)
        # self.modulation.bias.data.fill_(1.0)
        self.scale = 1.0 / math.sqrt(channels_in * kernel_size ** 2)
        # self.scale = 1.0 / math.sqrt(channels_in)
        # self.convdw = nn.utils.weight_norm(nn.Conv2d(channels_in, channels_in, kernel_size, padding=kernel_size//2, groups=channels_in, bias = False))
        # self.convpw = nn.utils.weight_norm(nn.Conv2d(channels_in, channels_out, 1, bias = False))
        self.convpw = nn.utils.weight_norm(nn.Conv2d(channels_in, channels_out, 1, bias = False))
        self.convdw = nn.utils.weight_norm(nn.Conv2d(channels_out, channels_out, kernel_size, padding=kernel_size//2, groups=channels_out, bias = False))
        # self.convpw = nn.Conv2d(channels_in, channels_out, 1, bias = False)
        # self.convdw = nn.Conv2d(channels_out, channels_out, kernel_size, padding=kernel_size//2, groups=channels_out, bias = False)
        nn.init.kaiming_normal_(self.convpw.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.convdw.weight, mode="fan_in", nonlinearity="leaky_relu")
        # if self.convpw.bias is not None:
        #     nn.init.constant_(self.convpw.bias, 0)
        # if self.convdw.bias is not None:
        #     nn.init.constant_(self.convdw.bias, 0)
        # nn.init.normal_(self.convpw.weight)
        # nn.init.normal_(self.convdw.weight)

    def forward(self, x, style):
        mstyle = self.modulation(style)*self.scale
        mstyle = mstyle.view(mstyle.size(0), -1, 1, 1)
        x = mstyle*x
        # x = self.convdw(x)
        # x = self.convpw(x)
        x = self.convpw(x)
        x = self.convdw(x)
        # x = self.mean_only_in(x)
        # x = self.scale*x
        return x
    # def forward(self, x, style):
    #     x = self.adain(x, style)
    #     x = self.convpw(x)
    #     x = self.convdw(x)
    #     return x

class WNConv2d_small_t(nn.Module):
    def __init__(
            self,
            channels_in,
            channels_out,
            style_dim,
            kernel_size,
            demodulate=True
    ):
        super().__init__()
        self.modulation1 = nn.Linear(style_dim, channels_in, bias=True)
        nn.init.kaiming_normal_(self.modulation1.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.modulation1.bias.data.fill_(1.0)
        self.modulation2 = nn.Linear(style_dim, channels_out, bias=True)
        nn.init.kaiming_normal_(self.modulation2.weight, mode="fan_in", nonlinearity="leaky_relu")
        self.modulation2.bias.data.fill_(1.0)
        self.scale1 = 1.0 / math.sqrt(channels_in)
        self.scale2 = 1.0 / math.sqrt(channels_out * kernel_size ** 2)
        self.ac = ScaledLeakyReLU()
        # self.convdw = nn.utils.weight_norm(nn.Conv2d(channels_in, channels_in, kernel_size, padding=kernel_size//2, groups=channels_in, bias = False))
        # self.convpw = nn.utils.weight_norm(nn.Conv2d(channels_in, channels_out, 1, bias = False))
        # self.convpw = nn.utils.weight_norm(nn.Conv2d(channels_in, channels_out, 1, bias = False))
        # self.convdw = nn.utils.weight_norm(nn.Conv2d(channels_out, channels_out, kernel_size, padding=kernel_size//2, groups=channels_out, bias = False))
        self.convpw = nn.Conv2d(channels_in, channels_out, 1, bias = False)
        self.convdw = nn.Conv2d(channels_out, channels_out, kernel_size, padding=kernel_size//2, groups=channels_out, bias = False)
        nn.init.kaiming_normal_(self.convpw.weight, mode="fan_in", nonlinearity="leaky_relu")
        nn.init.kaiming_normal_(self.convdw.weight, mode="fan_in", nonlinearity="leaky_relu")
        # nn.init.normal_(self.convpw.weight)
        # nn.init.normal_(self.convdw.weight)

    def forward(self, x, style):
        mstyle1 = self.modulation1(style)*self.scale1
        mstyle1 = mstyle1.view(mstyle1.size(0), -1, 1, 1)
        x = mstyle1*x
        x = self.ac(self.convpw(x))

        mstyle2 = self.modulation2(style)*self.scale2
        mstyle2 = mstyle2.view(mstyle2.size(0), -1, 1, 1)
        x = mstyle2*x
        x = self.convdw(x)
        return x

class WNStyleConv2d(nn.Module):
    def __init__(
            self,
            style_dim,
            channels_in,
            channels_out,
            kernel_size,
            stride=1, 
            padding=0, 
            bias=True, 
            groups=1,
            weight_norm = True
    ):
        super().__init__()
        if style_dim > 0:
            # self.modulation = nn.Linear(style_dim, channels_in, bias=False)
            self.modulation = nn.Sequential(
                nn.Linear(style_dim, channels_in, bias=False),
                # nn.Linear(channels_in, channels_in, bias=False)
            )
            for m in self.modulation.modules():
                if isinstance(m, nn.Linear):
                    nn.init.uniform_(m.weight, -0.02, 0.02)
                    # nn.init.normal_(m.weight)
            # nn.init.kaiming_normal_(self.modulation.weight, mode="fan_in", nonlinearity="leaky_relu")
            # nn.init.kaiming_uniform_(self.modulation.weight, mode="fan_in", nonlinearity="leaky_relu")
            # nn.init.normal_(self.modulation.weight)
            # nn.init.uniform_(self.modulation.weight, -0.02, 0.02)
            # nn.init.uniform_(self.modulation.weight, -0.2, 0.2)

        # self.modulation_scale = nn.Parameter( torch.ones(1)*(1.0 / math.sqrt(channels_in)) )
        self.modulation_scale = nn.Parameter( torch.ones(1)*0.01 )
        self.bias_scale = nn.Parameter( torch.ones(1) )
        # self.conv_scale = nn.Parameter( torch.ones(1)*0.5 )
        # self.sigmoid = nn.Hardsigmoid()
        # self.sigmoid = nn.Sigmoid()
        # self.sigmoid_scalex = nn.Parameter( torch.ones(1)*2.0 )
        # self.sigmoid_scaley = nn.Parameter( torch.ones(1)*2.0 )
        self.inc = channels_in
        

        if weight_norm:
            self.conv = nn.utils.weight_norm(nn.Conv2d(channels_in, channels_out, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups))
        else:
            self.conv = nn.Conv2d(channels_in, channels_out, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)
        nn.init.kaiming_normal_(self.conv.weight, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x, style):
        if self.modulation != None:
            mstyle =  self.bias_scale + self.modulation(style) * self.modulation_scale
            # mstyle = 1. + self.sigmoid_scaley*(self.sigmoid(self.modulation(style)/self.sigmoid_scalex) - 0.5)
            # print("--------inc:", self.inc)
            # print("self.sigmoid_scalex:", self.sigmoid_scalex)
            # print("self.sigmoid_scaley:", self.sigmoid_scaley)
            # print("self.modulation_scale:", self.modulation_scale)
            # print("style:", mstyle)
            mstyle = mstyle.view(mstyle.size(0), -1, 1, 1)
            x = mstyle*x
        x = self.conv(x)
        # x = x * self.conv_scale
        return x

class StyledResBlock(nn.Module):
    def __init__(self, in_channel, style_dim, kernel_size, demodulate=True):
        super().__init__()

        # self.conv1 = ModulatedConv2d(in_channel, in_channel, style_dim, 3)
        # self.conv2 = ModulatedConv2d(in_channel, in_channel, style_dim, 3)

        self.conv1 = StyledConv2d(in_channel, in_channel//2, style_dim, kernel_size)
        self.conv2 = ModulatedConv2d(in_channel//2, in_channel, style_dim, kernel_size)

        # self.conv1 = StyledConv(in_channel, in_channel*2, 1, style_dim, upsample=False, blur_kernel=blur_kernel, demodulate=demodulate)
        # self.conv2 = StyledConv(in_channel*2, in_channel*2, 3, style_dim, upsample=False, blur_kernel=blur_kernel, demodulate=demodulate, group=in_channel)
        # self.conv3 = StyledConv(in_channel*2, in_channel, 1, style_dim, upsample=False, blur_kernel=blur_kernel, demodulate=demodulate, activation=False)

        # self.conv1 = StyledConv(in_channel, in_channel//2, 3, style_dim, upsample=False, blur_kernel=blur_kernel, demodulate=demodulate)
        # self.conv2 = StyledConv(in_channel//2, in_channel, 3, style_dim, upsample=False, blur_kernel=blur_kernel, demodulate=demodulate)

    def forward(self, input, style):
        out = self.conv1(input, style)
        out = self.conv2(out, style)
        # out = self.conv3(out, style)
        out = (out + input) / math.sqrt(2)

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

class StyledConv2d(nn.Module):
    def __init__(
        self,
        channels_in,
        channels_out,
        style_dim,
        kernel_size,
        demodulate=True,
        conv_module=ModulatedConv2d
    ):
        super().__init__()

        self.conv = conv_module(
            channels_in,
            channels_out,
            style_dim,
            kernel_size,
            demodulate=demodulate
        )

        self.bias = nn.Parameter(torch.zeros(1, channels_out, 1, 1))
        self.act = ScaledLeakyReLU()

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.act(out + self.bias)
        return out

class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, kernel_size, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv = ModulatedConv2d(in_channel, 4, style_dim, kernel_size, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 4, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out

class StyledConv2dUpsample(nn.Module):
    def __init__(
        self,
        channels_in,
        channels_out,
        style_dim,
        kernel_size,
        demodulate=True
    ):
        super().__init__()

        # self.conv1 = conv_module(
        #     channels_in,
        #     channels_out,
        #     style_dim,
        #     1,
        #     demodulate=demodulate
        # )

        # self.conv1 = ModulatedConv2d(channels_in, channels_out, style_dim, 1, demodulate=demodulate)

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # self.conv2 = ModulatedConv2d(
        #     channels_out,
        #     channels_out,
        #     style_dim,
        #     3,
        #     demodulate=demodulate
        # )
        self.conv2 = ModulatedConv2d(
            channels_in,
            channels_out,
            style_dim,
            kernel_size,
            demodulate=demodulate
        )

        self.bias = nn.Parameter(torch.zeros(1, channels_out, 1, 1))
        self.act = ScaledLeakyReLU()

        #self.conv3 = ModulatedConv2dcr(channels_out, channels_out, style_dim, 1, demodulate=demodulate)
        # self.conv3 = conv_module(
        #     channels_out,
        #     channels_out,
        #     style_dim,
        #     1,
        #     demodulate=demodulate
        # )

    def forward(self, input, style, noise=None):
        # out = self.conv1(input, style)
        out = self.up(input)
        out = self.conv2(out, style)
        out = self.act(out + self.bias)

        return out

class hsigmoid(nn.Module):
    def __init__(self):
        super(hsigmoid, self).__init__()
        self.hs = nn.Hardsigmoid()
    def forward(self, x):
        out = self.hs(x)
        return out
class SeModule(nn.Module):
    def __init__(self, in_size, reduction=2):
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

class SeWeight(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeWeight, self).__init__()
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
        return self.se(x)

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
        mlp = [PixelNorm(),           #this pixl norm make input latend to unit vector(avoid node too large or too small)
               nn.Linear(inc, dim),
               nn.LayerNorm([dim]),      #layer norm make training stable, or yon can try weight norm
               ActFunc,]
        for i in range(n_layers-1):
            mlp.extend([
                nn.Linear(dim, dim),
                nn.LayerNorm([dim]),
                ActFunc,
            ])
        mlp.extend([nn.Linear(dim, dim),])
        mlp.extend([PixelNorm(),])       #pixl norm make style stable(unit vector) thus make training stable
        self.mlp = nn.Sequential(*mlp)

        self.initialize_module(self)

    def forward(self, x):
        b, c = x.size(0), x.size(1)
        x = x.view(b, c)
        x = self.mlp(x)
        return x

    def initialize_module(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                # nn.init.uniform_(m.weight, -0.02, 0.02)
                nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
                # nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu")
                # nn.init.normal_(m.weight)
                if m.bias is not None:
                        nn.init.constant_(m.bias, 0)


channels2 = {
    16: 512,
    32: 128,
    64: 64,
    128: 32,
    256: 16,
}
out_channels = {
    0: 128,
    1: 64,
    2: 32,
    3: 16,
}

# class ResBlock(nn.Module):
#     def __init__(self, style_dim, in_channel, semodule = False):
#         super().__init__()

#         self.conv1 = WNStyleConv2d(style_dim, in_channel, in_channel, 1, weight_norm=False)
#         self.lrelu1 = ScaledLeakyReLU()
#         self.conv2 = WNStyleConv2d(style_dim, in_channel, in_channel, 5, padding=2, groups=in_channel)
#         # self.lrelu2 = ScaledLeakyReLU()
#         # self.conv3 = WNStyleConv2d(style_dim, expand_channel, in_channel, 1, weight_norm=False)

#         if semodule:
#             self.se = SeModule(in_channel)
#         else:
#             self.se = None   

#     def forward(self, input, style):
#         out = self.lrelu1(self.conv1(input, style))
#         out = self.conv2(out, style)
#         if self.se != None:
#             out = self.se(out)
#         out = (out + input) / math.sqrt(2)
#         return out

class ResBlock(nn.Module):
    def __init__(self, style_dim, in_channel, semodule = False):
        super().__init__()

        self.conv1 = WNStyleConv2d(style_dim, in_channel, in_channel, 5, padding=2, groups=in_channel)
        self.lrelu1 = ScaledLeakyReLU()
        self.conv2 = WNStyleConv2d(style_dim, in_channel, in_channel, 1, weight_norm=False)

        if semodule:
            self.sew = SeWeight(in_channel)
        else:
            self.sew = None   

    def forward(self, input, style):
        out = self.lrelu1(self.conv1(input, style))
        if self.sew != None:
            sew = self.sew(input)
            out = out * sew
        out = (out + input) / math.sqrt(2)
        out = self.conv2(out, style)
        return out

class ChannelsDownBlock(nn.Module):
    def __init__(self, style_dim, in_channel, out_channel):
        super().__init__()

        self.conv1 = WNStyleConv2d(style_dim, in_channel, in_channel, 5, padding=2, groups=in_channel)
        self.sew1 = SeWeight(in_channel) 
        self.lrelu1 = ScaledLeakyReLU()
        self.convp1 = WNStyleConv2d(style_dim, in_channel, 256, 1, weight_norm=False)

        self.conv2 = WNStyleConv2d(style_dim, 256, 256, 5, padding=2, groups=256)
        self.sew2 = SeWeight(256) 
        self.lrelu2 = ScaledLeakyReLU() 
        self.convp2 = WNStyleConv2d(style_dim, 256, out_channel, 1, weight_norm=False)

        self.conv3 = WNStyleConv2d(style_dim, out_channel, out_channel, 5, padding=2, groups=out_channel)
        self.sew3 = SeWeight(out_channel) 
        self.lrelu3 = ScaledLeakyReLU() 
        # self.convp3 = WNStyleConv2d(style_dim, 128, 128, 1, weight_norm=False)

    def forward(self, input, style):
        sew1 = self.sew1(input)
        out = self.lrelu1(sew1 * self.conv1(input, style))
        out = self.convp1(out, style)
            
        sew2 = self.sew2(out)
        out = self.lrelu2(sew2 * self.conv2(out, style))
        out = self.convp2(out, style)

        sew3 = self.sew3(out)
        out = self.lrelu3(sew3 * self.conv3(out, style))

        return out

class UpBlock(nn.Module):
    def __init__(self, style_dim, in_channel, out_channel, semodule = False):
        super().__init__()

        self.conv1 = WNStyleConv2d(style_dim, in_channel, out_channel, 1, weight_norm=False)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        # self.lrelu1 = ScaledLeakyReLU()
        self.conv2 = WNStyleConv2d(style_dim, out_channel, out_channel, 3, padding=1, groups=out_channel)
        self.lrelu2 = ScaledLeakyReLU()
        # self.conv3 = WNStyleConv2d(expand_channel, in_channel, 1, weight_norm=False)

        if semodule:
            self.se = SeModule(in_channel)
        else:
            self.se = None

    def forward(self, input, style):
        out = self.up(self.conv1(input, style))
        out = self.lrelu2(self.conv2(out, style))

        if self.se != None:
            out = self.se(out)
        return out

class RGBABlock(nn.Module):
    def __init__(self, style_dim, in_channel, out_channel):
        super().__init__()

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = WNStyleConv2d(style_dim, in_channel, out_channel, 5, padding=2, groups=out_channel, weight_norm=True)

    def forward(self, input, rgba, style):
        rgba = self.up(rgba)
        out = self.conv(input, style)
        out = rgba + out
 
        return out

class TinyGeneratortttt(nn.Module):
    def __init__(
        self
    ):
        super(TinyGeneratortttt, self).__init__()

        self.latent_dim = 8
        self.style_dim = 512
        self.n_mlp = 5

        self.input_nc = 512
        self.output_nc = 3
        self.size = 256
        self.num_down = 4
        self.n_mlp = 5

        style_dim = 512
        no_style = -1
        in_channel = 512

        # MLP
        self.mapping = MLP(self.latent_dim, style_dim, self.n_mlp)
        
        self.resblock = nn.ModuleList([
            ResBlock(style_dim, 512, semodule=True),
            ResBlock(style_dim, 512, semodule=True),
        ])

        self.downcblock = ChannelsDownBlock(style_dim, 512, 128)

        self.to_rgb = WNStyleConv2d(style_dim, 512, 4, 5, padding=2, groups=4)
        self.to_alpha = hsigmoid()

        out_channels = {
            0: 64,
            1: 32,
            2: 16,
            3: 8,
        }
        self.convs = nn.ModuleList([
            UpBlock(style_dim, 128, out_channels[0]),
            UpBlock(style_dim, out_channels[0], out_channels[1]),
            UpBlock(style_dim, out_channels[1], out_channels[2]),
            UpBlock(style_dim, out_channels[2], out_channels[3]),
        ])
        self.to_rgbas = nn.ModuleList([
            RGBABlock(style_dim, out_channels[0], 4),
            RGBABlock(style_dim, out_channels[1], 4),
            RGBABlock(style_dim, out_channels[2], 4),
            RGBABlock(style_dim, out_channels[3], 4),
        ])

    def forward(self, input, styles):
        styles = self.mapping(styles)

        out = input
        for resconv in self.resblock:
            out = resconv(out, styles)

        rgba = self.to_rgb(out, styles)

        for conv, to_rgba in zip(
            self.convs, self.to_rgbas
        ):
            out = conv(out, styles)
            rgba = to_rgba(out, rgba, styles)

        image = rgba[:,0:3,:,:]
        alpha = self.to_alpha(rgba[:,3:4,:,:])

        return image, alpha

class TinyGenerator(nn.Module):
    def __init__(
        self
    ):
        super(TinyGenerator, self).__init__()

        self.latent_dim = 8
        self.style_dim = 512
        self.n_mlp = 5

        self.input_nc = 512
        self.output_nc = 3
        self.size = 256
        self.num_down = 4
        self.n_mlp = 5

        style_dim = 512
        no_style = -1
        in_channel = 512

        # MLP
        self.mapping = MLP(self.latent_dim, style_dim, self.n_mlp)
        
        # self.resblock = nn.ModuleList([
        #     ResBlock(style_dim, 512, semodule=True),
        #     ResBlock(style_dim, 512, semodule=True),
        # ])

        self.downcblock = ChannelsDownBlock(style_dim, 512, 256)

        self.to_rgb = WNStyleConv2d(style_dim, 256, 4, 5, padding=2, groups=4)
        self.to_alpha = hsigmoid()

        # out_channels = {
        #     0: 64,
        #     1: 32,
        #     2: 16,
        #     3: 8,
        # }
        out_channels = {
            0: 128,
            1: 64,
            2: 32,
            3: 16,
        }
        self.convs = nn.ModuleList([
            UpBlock(style_dim, 256, out_channels[0]),
            UpBlock(style_dim, out_channels[0], out_channels[1]),
            UpBlock(style_dim, out_channels[1], out_channels[2]),
            UpBlock(style_dim, out_channels[2], out_channels[3]),
        ])
        self.to_rgbas = nn.ModuleList([
            RGBABlock(style_dim, out_channels[0], 4),
            RGBABlock(style_dim, out_channels[1], 4),
            RGBABlock(style_dim, out_channels[2], 4),
            RGBABlock(style_dim, out_channels[3], 4),
        ])

    def forward(self, input, styles):
        styles = self.mapping(styles)

        # out = input
        # for resconv in self.resblock:
        #     out = resconv(out, styles)

        out = self.downcblock(input, styles)

        rgba = self.to_rgb(out, styles)

        for conv, to_rgba in zip(
            self.convs, self.to_rgbas
        ):
            out = conv(out, styles)
            rgba = to_rgba(out, rgba, styles)

        image = rgba[:,0:3,:,:]
        alpha = self.to_alpha(rgba[:,3:4,:,:])

        return image, alpha


from matplotlib import pyplot as plt 
from skimage import io
from skimage import transform
import numpy as np
from torchvision import transforms
from thop import profile

totensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), inplace=True)
    ])

def toNumpy(img_tensor):
    out = torch.reshape(img_tensor, (img_tensor.shape[1], img_tensor.shape[2], img_tensor.shape[3]))
    out = out.detach().numpy().transpose((1, 2, 0))
    return out

def toNumpyWithNorm(img_tensor):
    out = torch.reshape(img_tensor, (img_tensor.shape[1], img_tensor.shape[2], img_tensor.shape[3]))
    out = out.detach().numpy().transpose((1, 2, 0))
    # out = out - np.min(out)/(np.max(out)-np.min(out))
    # out = normalize(out)*0.5+0.5
    out = out*0.5+0.5
    return out

def normalize(x):
    mean = np.mean(x[:])
    std = np.std(x[:])
    return (x-mean)/std

def test():
    torch.set_printoptions(precision=4, sci_mode=False)
    net = TinyGenerator()
    
    style = torch.randn(1, 8)
    style2 = torch.randn(1, 8)

    # type1
    rand_x = torch.randn(1,512,16,16)

    # type2
    img = io.imread("/Users/cr/git/face/Morph-UGATIT/datasets/trainA/female_4493.jpg")
    # img_tensortt = totensor(img)
    # print(torch.min(img_tensortt))
    # print(torch.max(img_tensortt))
    img = transform.resize(img, (16, 16))
    img = normalize(img)
    img_tensor = torch.from_numpy(img.transpose((2,0,1)))
    img_tensor = torch.reshape(img_tensor, (1, 3, 16, 16))[:,0:2,:,:]
    img_x = img_tensor.repeat((1, 256, 1, 1)).float()

    rand_x_out, _ = net(rand_x, style)
    img_x_out, _ = net(img_x, style)
    rand_x_out2, _ = net(rand_x, style2)
    img_x_out2, _ = net(img_x, style2)

    rand_x_in = toNumpy(rand_x[0:1,0:1,:,:])
    rand_x_out = toNumpyWithNorm(rand_x_out)
    rand_x_out2 = toNumpyWithNorm(rand_x_out2)
    img_x_in = toNumpy(img_x[0:1,1:2,:,:])
    img_x_out = toNumpyWithNorm(img_x_out)
    img_x_out2 = toNumpyWithNorm(img_x_out2)

    million = 100 * 10000
    flops256, _ = profile(net, (img_x,style,))
    print("decoder flops:", flops256/million)

    plt.figure("haha")
    plt.subplot(2,3,1), plt.title('rand_x_in')
    plt.imshow(rand_x_in)
    plt.subplot(2,3,2), plt.title('rand_x_out')
    plt.imshow(rand_x_out)
    plt.subplot(2,3,3), plt.title('rand_x_out2')
    plt.imshow(rand_x_out2)

    plt.subplot(2,3,4), plt.title('img_x_in')
    plt.imshow(img_x_in)
    plt.subplot(2,3,5), plt.title('img_x_out')
    plt.imshow(img_x_out)
    plt.subplot(2,3,6), plt.title('img_x_out2')
    plt.imshow(img_x_out2)

    plt.show()

import onnxruntime as rt
import time
def toonnx():

    net = TinyGenerator()
    
    rand_x = torch.randn(1,512,16,16)
    style = torch.randn(1, 8)

    origin, _ = net(rand_x, style)

    export_onnx_file = "decoder.onnx"
    torch.onnx.export(
        net,
        (rand_x,style),
        export_onnx_file,
        opset_version=11,
        do_constant_folding=True,
        input_names=["x", "style"],
        output_names=["output"],
        training = torch.onnx.TrainingMode.EVAL,
        # verbose=True,
        # operator_export_type=torch.onnx.OperatorExportTypes.ONNX_FALLTHROUGH,
        # dynamic_axes={"input":{0:"batch_size",2:"batch_size",3:"batch_size"}, "output":{0:"batch_size"}}
        )

    rand_x_in = rand_x.detach().cpu().numpy()
    rand_x_in = rand_x_in.astype(np.float32)
    style = style.detach().cpu().numpy()
    style = style.astype(np.float32)
    sess = rt.InferenceSession(export_onnx_file)

    start_time = time.time()
    out = sess.run([], {
            'x': rand_x_in,
            'style': style
        })
    elapse_time = time.time() - start_time
    print(elapse_time)
    
    origin = toNumpyWithNorm(origin)
    out = np.reshape(out[0], (3, 256, 256)).transpose((1, 2, 0))
    out = out*0.5+0.5

    plt.figure("kkk")
    plt.subplot(1,2,1), plt.title('origin')
    plt.imshow(origin)
    plt.subplot(1,2,2), plt.title('onnx')
    plt.imshow(out)
    plt.show()

# test()
# toonnx()