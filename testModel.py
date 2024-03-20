import numpy as np
import os
import torch

from ModelPreparation.models.model import RefineNet

x = torch.randn((1,280,480))
out = RefineNet(x)
print(out.shape)
exit()