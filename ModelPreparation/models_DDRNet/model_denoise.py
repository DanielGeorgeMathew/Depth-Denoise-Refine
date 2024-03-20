import torch
from torch import nn


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.LeakyReLU(inplace=True)
    )


def conv(in_planes, out_planes, k_size, pad, stride):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=k_size, padding=pad, stride=stride),
        nn.LeakyReLU(inplace=True)
    )


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out


class convResnet(nn.Module):
    def __init__(self, ngf=32, n_blocks=2, n_down=2, learn_residual=True, use_color=False):
        super(convResnet, self).__init__()
        self.use_color = use_color
        self.n_down = n_down
        self.n_blocks = n_blocks
        self.learn_residual = learn_residual
        # if use_color:
        #     self.conv0_d = conv(1, ngf // 2, 3, 1, 1)
        #     self.conv0_c = conv(3, ngf // 2, 3, 1, 1)
        # else:
        self.conv0 = conv(1, ngf, 3, 1, 1)
        channels = [ngf]
        mult = 1
        in_channels = ngf

        self.encoder = nn.ModuleList([])
        for i in range(n_down):
            mult *= 2
            self.encoder.append(conv(in_channels, ngf * mult, k_size=3, pad=1, stride=2))
            channels.append(ngf * mult)
            in_channels = ngf * mult

        self.residual_block = nn.ModuleList([])
        for i in range(n_blocks):
            self.residual_block.append(ResidualBlock(in_channels, in_channels))

        self.decoder = nn.ModuleList([])
        for i in range(n_down):
            self.decoder.append(upconv(channels[n_down - i] * 2, channels[n_down - i - 1]))

        # self.decoder = nn.ModuleList([])
        # for i in range(n_down):
        #     self.decoder.append(conv(channels[n_down - i] * 2, channels[n_down - i - 1], 3, 1, 1))

        self.final_conv = nn.Sequential(nn.Conv2d(ngf*2, 1, kernel_size=3, padding=1, stride=1),
                                        nn.Tanh())

    def forward(self, x, rgb=None):
        scale_skips = []
        residual = x
        # if self.use_color:
        #     out_d = self.conv0_d(x)
        #     out_c = self.conv0_c(rgb)
        #     out = torch.cat([out_d, out_c], dim=1)
        # else:
        out = self.conv0(x)
        scale_skips.append(out)
        for i in range(self.n_down):
            out = self.encoder[i](out)
            scale_skips.append(out)

        for i in range(self.n_blocks):
            out = self.residual_block[i](out)

        for i in range(self.n_down):
            out = torch.cat([out, scale_skips[-1]], dim=1)
            scale_skips.pop()
            out = self.decoder[i](out)

        # for i in range(self.n_down):
        #     out = torch.cat([out, scale_skips[-1]], dim=1)
        #     h, w = out.shape[2:]
        #     out = torch.nn.functional.interpolate(out, size=[2 * h, 2 * w], mode='bilinear')
        #     scale_skips.pop()
        #     out = self.decoder[i](out)

        out = torch.cat([out, scale_skips[0]], dim=1)
        assert len(scale_skips) == 1
        del scale_skips

        out = self.final_conv(out)

        if self.learn_residual:
            out = residual + out

        return out


# depth = torch.randn((6, 1, 320, 240))
# rgb = torch.randn((6, 3, 320, 240))
# model = convResnet(use_color=False)
# out = model(depth)
# print(out.shape)

# x = torch.randn((6,1,320,240))
# model = convResnet()
# out = model(x)
# print(out.shape)
