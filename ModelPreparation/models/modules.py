import torch
from torch import nn


class ConvRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvRelu, self).__init__()
        self.conv_relu = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
                                       nn.ReLU(inplace=True))
    def forward(self, x):
        return self.conv_relu(x)


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(
            *[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning


class Encoder(nn.Module):
    def __init__(self, num_features, growth_rate, num_blocks, num_layers):
        super(Encoder, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        # defining depth branch modules
        self.conv1_d = ConvRelu(1, num_features, kernel_size=3, padding=3 // 2)

        self.rdbs_d = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs_d.append(RDB(self.G, self.G, self.C))

        self.conv2_d = ConvRelu(self.G * self.D, self.G0, kernel_size=3, padding=3 // 2)

        self.shallow_fusion = ConvRelu(self.G0 * 2, self.G, kernel_size=3, padding=3 // 2)
        self.feature_fusion_list = nn.ModuleList([ConvRelu(self.G * 2, self.G, kernel_size=3, padding=3 // 2)] * self.D)

        # defining image branch modules
        self.conv1_c = ConvRelu(3, num_features, kernel_size=3, padding=3 // 2)

        self.rdbs_c = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs_c.append(RDB(self.G, self.G, self.C))

        self.conv2_c = ConvRelu(self.G * self.D, self.G0, kernel_size=3, padding=3 // 2)

    def forward(self, img, depth):
        conv1_features_c = self.conv1_c(img)

        local_features_c = []
        x = conv1_features_c
        for i in range(self.D):
            x = self.rdbs_c[i](x)
            local_features_c.append(x)
        out_c = self.conv2_c(torch.cat(local_features_c, 1)) + conv1_features_c

        conv1_features_d = self.conv1_d(depth)

        local_features_d = []
        x = self.shallow_fusion(torch.cat([conv1_features_d, conv1_features_c], 1))
        for i in range(self.D):
            x = self.rdbs_d[i](x)
            local_features_d.append(x)
            x = self.feature_fusion_list[i](torch.cat([x, local_features_c[i]], 1))

        out_d = self.conv2_d(torch.cat(local_features_d, 1)) + conv1_features_d

        return out_c, out_d


class Decoder(nn.Module):
    def __init__(self, num_features, growth_rate, num_blocks, num_layers):
        super(Decoder, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers

        self.conv3 = ConvRelu(64, num_features, kernel_size=3, padding=3 // 2)

        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))

        self.conv4 = ConvRelu(self.G * self.D, self.G0, kernel_size=3, padding=3 // 2)

        self.conv5 = ConvRelu(self.G, 1, kernel_size=1, padding=0)

    def forward(self, x):
        conv3_features = self.conv3(x)

        local_features = []
        x = conv3_features
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        out = self.conv5(self.conv4(torch.cat(local_features, 1)) + conv3_features)

        return out

# im = torch.randn((1, 3, 280, 480))
# depth = torch.randn((1, 1, 280, 480))
# encoder = Encoder(32, 32, 3, 3)
# img_features, depth_features = encoder(im, depth)
# print(img_features.shape)
# print(depth_features.shape)

# class RGBFeatureExtractor(nn.Module):
#     def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_blocks, num_layers):
#         super(RGBFeatureExtractor, self).__init__()
#         self.G0 = num_features
#         self.G = growth_rate
#         self.D = num_blocks
#         self.C = num_layers
#
#         # shallow feature extraction
#         self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
#         self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)
#
#         # residual dense blocks
#         self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
#         for _ in range(self.D - 1):
#             self.rdbs.append(RDB(self.G, self.G, self.C))
#
#         # global feature fusion
#         self.gff = nn.Sequential(
#             nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
#             nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2))
#
#         # up-sampling
#         assert 2 <= scale_factor <= 4
#         if scale_factor == 2 or scale_factor == 4:
#             self.upscale = []
#             for _ in range(scale_factor // 2):
#                 self.upscale.extend([nn.Conv2d(self.G0, self.G0 * (2 ** 2), kernel_size=3, padding=3 // 2),
#                                      nn.PixelShuffle(2)])
#             self.upscale = nn.Sequential(*self.upscale)
#         else:
#             self.upscale = nn.Sequential(
#                 nn.Conv2d(self.G0, self.G0 * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
#                 nn.PixelShuffle(scale_factor)
#             )
#
#         self.output = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)
#
#     def forward(self, x):
#         sfe1 = self.sfe1(x)
#         sfe2 = self.sfe2(sfe1)
#
#         x = sfe2
#         local_features = []
#         for i in range(self.D):
#             x = self.rdbs[i](x)
#             local_features.append(x)
#
#         x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
#         x = self.upscale(x)
#         x = self.output(x)
#         return x


# class DepthFeatureExtractor(nn.Module):
#     def __init__(self, scale_factor, num_channels, num_features, growth_rate, num_blocks, num_layers):
#         super(DepthFeatureExtractor, self).__init__()
#         self.G0 = num_features
#         self.G = growth_rate
#         self.D = num_blocks
#         self.C = num_layers
#
#         # shallow feature extraction
#         self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
#         self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)
#
#         # residual dense blocks
#         self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
#         for _ in range(self.D - 1):
#             self.rdbs.append(RDB(self.G, self.G, self.C))
#
#         # global feature fusion
#         self.gff = nn.Sequential(
#             nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
#             nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
#         )
#         if scale_factor == 2 or scale_factor == 4:
#             self.upscale = []
#
#         # up-sampling
#         assert 2 <= scale_factor <= 4
#         if scale_factor == 2 or scale_factor == 4:
#             self.upscale = []
#             for _ in range(scale_factor // 2):
#                 self.upscale.extend([nn.Conv2d(self.G0, self.G0 * (2 ** 2), kernel_size=3, padding=3 // 2),
#                                      nn.PixelShuffle(2)])
#             self.upscale = nn.Sequential(*self.upscale)
#         else:
#             self.upscale = nn.Sequential(
#                 nn.Conv2d(self.G0, self.G0 * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
#                 nn.PixelShuffle(scale_factor)
#             )
#         self.output = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)
#
#     def forward(self, x):
#         sfe1 = self.sfe1(x)
#         sfe2 = self.sfe2(sfe1)
#
#         x = sfe2
#         local_features = []
#         for i in range(self.D):
#             x = self.rdbs[i](x)
#             local_features.append(x)
#
#         x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
#         x = self.upscale(x)
#         x = self.output(x)
#         return x
