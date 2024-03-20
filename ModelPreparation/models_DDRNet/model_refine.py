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


class hyperColumn(nn.Module):
    def __init__(self, ngf=16):
        super(hyperColumn, self).__init__()
        self.conv1_d = conv(in_planes=1, out_planes=ngf // 2, k_size=3, pad=1, stride=1)
        self.conv1_c = conv(in_planes=3, out_planes=ngf // 2, k_size=3, pad=1, stride=1)
        self.conv2 = conv(in_planes=ngf, out_planes=ngf, k_size=3, pad=1, stride=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = conv(in_planes=ngf * 2, out_planes=ngf * 2, k_size=3, pad=1, stride=1)
        self.conv4 = conv(in_planes=ngf * 2, out_planes=ngf * 2, k_size=3, pad=1, stride=1)

        self.conv5 = conv(in_planes=ngf * 4, out_planes=ngf * 4, k_size=3, pad=1, stride=1)
        self.conv6 = conv(in_planes=ngf * 4, out_planes=ngf * 4, k_size=3, pad=1, stride=1)
        self.conv7 = conv(in_planes=ngf * 4, out_planes=ngf * 4, k_size=3, pad=1, stride=1)

        self.conv8 = conv(in_planes=ngf * 7, out_planes=ngf, k_size=1, pad=0, stride=1)
        self.conv9 = conv(in_planes=ngf, out_planes=ngf // 4, k_size=1, pad=0, stride=1)
        self.out = nn.Sequential(
            nn.Conv2d(in_channels=ngf // 4, out_channels=1, kernel_size=1, padding=0, stride=1),
            nn.Tanh())

        self.up1 = upconv(32, 32)
        self.up2 = upconv(64, 64)
        self.up3 = upconv(64, 64)

    def forward(self, depth, color):
        cnv1_d = self.conv1_d(depth)  # Bx8x320x240
        cnv1_c = self.conv1_c(color)  # Bx8x320x240
        cnv1 = torch.cat([cnv1_d, cnv1_c], dim=1)  # Bx16x320x240
        cnv2 = self.conv2(cnv1)  # Bx16x320x240
        hyper1 = torch.cat([cnv1, cnv2], dim=1)  # Bx32x320x240

        p_1 = self.pool(hyper1)  # Bx32x160x120
        cnv3 = self.conv3(p_1)  # Bx32x160x120
        cnv4 = self.conv4(cnv3)  # Bx32x160x120
        # up1 = torch.nn.functional.interpolate(cnv4, size=cnv2.shape[2:], mode='bilinear')  # Bx32x320x240
        up1 = self.up1(cnv4)

        hyper2 = torch.cat([cnv3, cnv4], dim=1)  # Bx64x160x120

        p_2 = self.pool(hyper2)  # Bx64x80x60
        cnv5 = self.conv5(p_2)  # Bx64x80x60
        cnv6 = self.conv6(cnv5)  # Bx64x80x60
        cnv7 = self.conv7(cnv6)  # Bx64x80x60
        # up2 = torch.nn.functional.interpolate(cnv7, size=cnv2.shape[2:], mode='bilinear')  # Bx64x320x240
        up2 = self.up2(cnv7)
        up2 = self.up3(up2)
        cat = torch.cat([cnv2, up1, up2], dim=1)  # Bx112x320x240

        cnv8 = self.conv8(cat)  # Bx16x320x240
        cnv9 = self.conv9(cnv8)  # Bx4x320x240
        out = self.out(cnv9)  # Bx1x320x240

        return out

# x = torch.randn((6, 1, 320, 240)).cuda()
# y = torch.randn((6, 3, 320, 240)).cuda()
# model = hyperColumn().cuda()
# out = model(x, y)
# print(out.shape)
