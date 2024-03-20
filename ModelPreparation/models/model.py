import torch
from torch import nn
from ModelPreparation.models.modules import Encoder, Decoder


# from modules import Encoder, Decoder


class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()
        self.encoder = Encoder(32, 32, 3, 3)
        self.decoder = Decoder(32, 32, 3, 3)

    def forward(self, img, depth):
        feat = self.encoder(img, depth)
        fused_feat = torch.cat(feat, 1)
        out = self.decoder(fused_feat)
        return out

# im = torch.randn((1, 3, 280, 480))
# depth = torch.randn((1, 1, 280, 480))
# model = RefineNet()
# out = model(im, depth)
# pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(out.shape)
# print(pytorch_total_params)
