import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ModulatedConv2dcr(nn.Module):
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


class ModulatedDWConv2dcr(nn.Module):
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

        # self.fconv1 = nn.Conv2d(channels_in, channels_in, kernel_size,groups=channels_in,padding=self.padding)
        # self.fconv2 = nn.Conv2d(channels_in, channels_out, 1)

    def forward(self, x, style):
        modulation = self.get_modulation(style)
        x = modulation * x
        x = F.conv2d(x, self.weight_dw, padding=self.padding, groups=x.size(1))
        x = F.conv2d(x, self.weight_permute)
        # x = self.fconv1(x)
        # x = self.fconv2(x)
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


class StyledResBlockcr(nn.Module):
    def __init__(self, in_channel, style_dim, blur_kernel=[1, 3, 3, 1], demodulate=True):
        super().__init__()

        self.conv1 = ModulatedDWConv2dcr(in_channel, in_channel, style_dim, 3)
        self.conv2 = ModulatedDWConv2dcr(in_channel, in_channel, style_dim, 3)

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


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        self.trace_model = False

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        if not hasattr(self, "noise") and self.trace_model:
            self.register_buffer("noise", noise)
        if self.trace_model:
            noise = self.noise
        return image + self.weight * noise

class StyledConv2dUpcr(nn.Module):
    def __init__(
        self,
        channels_in,
        channels_out,
        style_dim,
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

        self.conv1 = ModulatedConv2dcr(channels_in, channels_out, style_dim, 1, demodulate=demodulate)

        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.scale = 1.0 / math.sqrt(channels_in)

        self.conv2 = ModulatedDWConv2dcr(
            channels_out,
            channels_out,
            style_dim,
            3,
            demodulate=demodulate
        )

        self.bias = nn.Parameter(torch.zeros(1, channels_out, 1, 1))
        self.act = nn.LeakyReLU(0.2)

        #self.conv3 = ModulatedConv2dcr(channels_out, channels_out, style_dim, 1, demodulate=demodulate)
        # self.conv3 = conv_module(
        #     channels_out,
        #     channels_out,
        #     style_dim,
        #     1,
        #     demodulate=demodulate
        # )

    def forward(self, input, style, noise=None):
        
        out = self.conv1(input, style)
        out = self.up(out)*self.scale
        out = self.conv2(out, style)
        # out = self.conv3(out, style)
        out = self.act(out + self.bias)
        # out = self.conv3(out, style)

        return out

class StyledConv2dcr(nn.Module):
    def __init__(
        self,
        channels_in,
        channels_out,
        style_dim,
        kernel_size,
        demodulate=True,
        conv_module=ModulatedDWConv2dcr
    ):
        super().__init__()

        self.conv = conv_module(
            channels_in,
            channels_out,
            style_dim,
            kernel_size,
            demodulate=demodulate
        )

        self.noise = NoiseInjection()
        self.bias = nn.Parameter(torch.zeros(1, channels_out, 1, 1))
        self.act = nn.LeakyReLU(0.2)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        out = self.act(out + self.bias)
        return out

class ToRGBcr(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv = ModulatedDWConv2dcr(in_channel, 4, style_dim, 3, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 4, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out