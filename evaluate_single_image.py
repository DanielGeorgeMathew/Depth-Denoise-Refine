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

import open3d as o3d

REF_W, REF_H = 3024, 4032


def getExtrinsicIntrinsicFromMetadata(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata_dict = json.load(f)
    extrinsic_list = np.asarray([i['matrix'] for i in metadata_dict['extrinsic']])
    intrinsic = np.asarray(metadata_dict['intrinsic']['matrix']).T
    intrinsic[0][0] = intrinsic[0][0] * 480 / REF_W
    intrinsic[1][1] = intrinsic[1][1] * 640 / REF_H
    intrinsic[0][2] = intrinsic[0][2] * 480 / REF_W
    intrinsic[1][2] = intrinsic[1][2] * 640 / REF_H

    return extrinsic_list, intrinsic


def convert_from_uvd(u, v, d, A):
    fx = A[0][0]
    fy = A[1][1]
    cx = A[0][2]
    cy = A[1][2]

    x_over_z = (u - cx) / fx
    y_over_z = (v - cy) / fy
    z = -d
    y = x_over_z * z
    x = y_over_z * z
    return x, y, z


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


def enhance_rgb(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl, a, b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Stacking the original image with the enhanced image
    result = np.hstack((img, enhanced_img))
    plt.imshow(result)
    plt.show()
    return enhanced_img


root = '~/Downloads/DDRNet-TrainData-v2/DDRNet-TrainData-v2/scan-28.xcappdata/AppData/tmp'
frame_id = 25
weights_path = '~/Daniel/DepthRefinement/checkpoints/DDRNet-v3'

up_thresh = 65535
low_thresh = 0
thresh_range = (up_thresh - low_thresh) / 2.0
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")

# gt_depth = cv2.imread(os.path.join(root, 'gt_depth/{}.png'.format(frame_id)), cv2.IMREAD_ANYDEPTH).astype(float)
rgb = cv2.cvtColor(cv2.imread(os.path.join(root, 'rgb/{}.png'.format(frame_id))), cv2.COLOR_BGR2RGB)
rgb_copy = np.copy(rgb)
# rgb = enhance_rgb(rgb)

depth = cv2.imread(os.path.join(root, 'depth/{}.png'.format(frame_id)), cv2.IMREAD_ANYDEPTH).astype(float)
depth_copy = np.copy(depth)

mask = np.ones_like(depth)
# mask[gt_depth == 0] = 0
mask[depth == 0] = 0
# plt.imshow(mask)
# plt.show()

# mask = np.ones_like(depth)
mask[depth > (250.0 + 256*2)*30] = 0
# # mask[gt_depth == 0] = 0
# # mask[depth == 0] = 0
plt.imshow(mask)
plt.show()

d_model = convResnet()
r_model = hyperColumn()
d_model.to(device)
r_model.to(device)

d_model.load_state_dict(torch.load(os.path.join(weights_path, 'denoise_model.pt'), map_location=device))
r_model.load_state_dict(torch.load(os.path.join(weights_path, 'refine_model.pt'), map_location=device))

d_model.eval()
r_model.eval()

rgb_tensor = rgb_to_tensor(rgb)
depth_tensor = depth_to_tensor(depth_copy*mask)

denoised_output = d_model(depth_tensor.unsqueeze(0).to(device))
refined_output = r_model(denoised_output, rgb_tensor.unsqueeze(0).to(device))
refined_output = refined_output.detach().cpu().numpy().squeeze()
denoised_output = denoised_output.detach().cpu().numpy().squeeze()
denoised_output = (denoised_output + 1) * thresh_range
refined_output = (refined_output + 1) * thresh_range
denoised_output = denoised_output * mask
refined_output = refined_output * mask

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
plt.close(fig)

if not os.path.exists(os.path.join(root, 'compare_pcd')):
    os.makedirs(os.path.join(root, 'compare_pcd'))
cv2.imwrite(os.path.join(root, 'compare_pcd/denoised_depth.png'), denoised_output.astype(np.uint16))
cv2.imwrite(os.path.join(root, 'compare_pcd/refined_depth.png'), refined_output.astype(np.uint16))
cv2.imwrite(os.path.join(root, 'compare_pcd/orig_depth.png'), depth.astype(np.uint16))

for i in ['orig', 'denoised', 'refined']:
    depth_path = os.path.join(root, 'compare_pcd/{}_depth.png'.format(i))
    depth_img = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH) / 30
    _, intrinsic = getExtrinsicIntrinsicFromMetadata(
        os.path.join(root, 'metadata.json'))
    pcd = o3d.geometry.PointCloud()
    xyz = []
    h, w = depth_img.shape

    for u in range(h):
        for v in range(w):
            xyz.append(convert_from_uvd(u, v, depth_img[u][v], intrinsic))

    xyz = np.asarray(xyz)

    pcd.points = o3d.utility.Vector3dVector(xyz)
    # print(np.unique(xyz))
    # print(xyz)
    # o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(
        os.path.join(root, 'compare_pcd/{}_depth.ply'.format(i)),
        pcd)
