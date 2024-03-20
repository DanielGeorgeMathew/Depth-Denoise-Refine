import argparse
import json
import os
import random

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torchvision.transforms import Compose

from ModelPreparation.models_DDRNet.model_denoise import convResnet
from ModelPreparation.models_DDRNet.model_refine import hyperColumn
from ModelPreparation.loss.loss_main import DDRNet_Loss
from ModelPreparation.data.datasets import sureScanDataset
from ModelPreparation.models.util import EarlyStopping
from ModelPreparation.data.transforms import Scale, RandomHorizontalFlip, CenterCrop, ToTensor, ColorJitter, Normalize, \
    RandomCrop


def depth_to_tensor(pic):
    high_thresh = 65535
    low_thresh = 0
    thresh_range = (high_thresh - low_thresh) / 2.0
    pic_array = ((np.asarray(pic) - low_thresh) / thresh_range).astype(np.float32) - 1
    # pic_array = (np.asarray(pic) / 65535).astype(np.float32)
    pic_tensor = torch.from_numpy(pic_array).unsqueeze(0)
    return pic_tensor


def rgb_to_tensor(pic):
    pic_array = (np.asarray(pic) / 127.0).astype(np.float32) - 1.0
    pic_tensor = torch.permute(torch.from_numpy(pic_array), [2, 0, 1])
    return pic_tensor
    # pic_tensor = torch.permute(torch.from_numpy(pic_array), [2, 0, 1])
    # _, rgb_edges = kornia.filters.canny(pic_tensor.unsqueeze(0))
    # return rgb_edges.squeeze().unsqueeze(0)


root = '~/Downloads/scan-16.xcappdata/scan-16.xcappdata/AppData/tmp'
# gt_depth = cv2.imread(
#     os.path.join(root, 'gt_depth/0.png'),
#     cv2.IMREAD_ANYDEPTH).astype(float)
# mask = np.ones_like(gt_depth)

rgb = cv2.cvtColor(
    cv2.imread(os.path.join(root, 'rgb/200.png')),
    cv2.COLOR_BGR2RGB)
rgb_copy = np.copy(rgb)
depth = cv2.imread(
    os.path.join(root, 'depth/200.png'),
    cv2.IMREAD_ANYDEPTH).astype(float)
depth_copy = np.copy(depth)
mask = np.ones_like(depth)
mask[depth<30000] = 0
plt.imshow(mask)
plt.show()
exit()
device = torch.device('cpu')
d_model = convResnet()
r_model = hyperColumn()
d_model.to(device)
r_model.to(device)

d_model.load_state_dict(
    torch.load('~/Daniel/DepthRefinement/checkpoints/DDRNet-Final-OfficeData/denoise_model.pt',
               map_location=device))
r_model.load_state_dict(
    torch.load('~/Daniel/DepthRefinement/checkpoints/DDRNet-Final-OfficeData/refine_model.pt',
               map_location=device))

rgb_tensor = rgb_to_tensor(rgb)
depth_tensor = depth_to_tensor(depth_copy)

sample = {'rgb': rgb_tensor, 'depth': depth_tensor, 'gt_depth': None}

up_thresh = 65535
low_thresh = 0
thresh_range = (up_thresh - low_thresh) / 2.0
denoised_output = d_model(depth_tensor.unsqueeze(0).to(device))
refined_output = r_model(torch.clamp(denoised_output, -1, 1), rgb_tensor.unsqueeze(0).to(device)).detach().cpu().numpy().squeeze()
denoised_output = denoised_output.detach().cpu().numpy().squeeze()
denoised_output = (denoised_output + 1) * thresh_range
refined_output = (refined_output + 1) * thresh_range
denoised_output = denoised_output*mask + depth*(1-mask)
refined_output = refined_output*mask + depth*(1-mask)
fontsize = 10
fig, ax = plt.subplots(1, 4, figsize=(15, 15))

ax[0].set_title("RGB", fontsize=fontsize)
ax[0].imshow(rgb_copy)

ax[1].set_title("DEPTH", fontsize=fontsize)
ax[1].imshow(depth_copy / 30000)

ax[2].set_title("DENOISED DEPTH", fontsize=fontsize)
ax[2].imshow(denoised_output / 30000)
ax[3].set_title("REFINED DEPTH", fontsize=fontsize)
ax[3].imshow(refined_output / 30000)
plt.show()
# writer.add_figure("RANDOM VALIDATION DURING TRAIN", fig, epoch)
plt.close(fig)
#
if not os.path.exists(os.path.join(root, 'compare_pcd')):
    os.makedirs(os.path.join(root, 'compare_pcd'))
cv2.imwrite(os.path.join(root, 'compare_pcd/denoised_depth.png'), denoised_output.astype(np.uint16))
cv2.imwrite(os.path.join(root, 'compare_pcd/refined_depth.png'), refined_output.astype(np.uint16))
cv2.imwrite(os.path.join(root, 'compare_pcd/orig_depth.png'), depth.astype(np.uint16))
