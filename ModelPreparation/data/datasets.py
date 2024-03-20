import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageFilter
import random
import json
import os
import cv2
from PIL import ImageOps
from ModelPreparation.data.transforms import Scale, RandomHorizontalFlip, CenterCrop, ToTensor, ColorJitter, Normalize, \
    RandomCrop
import matplotlib.pyplot as plt
import torch

# __imagenet_stats = {'mean': [0.485, 0.456, 0.406],
#                     'std': [0.229, 0.224, 0.225]}
#
# train_transforms = transforms.Compose([
#     RandomHorizontalFlip(),
#     ToTensor(),
#     Normalize(__imagenet_stats['mean'],
#               __imagenet_stats['std'])
# ])
#
# test_transforms = transforms.Compose([
#     ToTensor(),
#     Normalize(__imagenet_stats['mean'],
#               __imagenet_stats['std'])
# ])
#
# all_transforms = {'train': train_transforms, 'val': test_transforms}


class sureScanDataset(Dataset):
    def __init__(self, file_paths, all_transforms):
        self.mode = os.path.basename(file_paths).split('.')[0]
        with open(file_paths, 'r') as fp:
            lines = fp.readlines()
        lines = np.asarray([i.rstrip().split(' ') for i in lines])

        self.rgb_paths = lines[:, 0]
        self.depth_paths = lines[:, 1]
        self.albedo_paths = lines[:, 2]
        self.gt_depth_paths = lines[:, 3]

        n = len(self.rgb_paths)
        print("Number of datapoints in {} set: {}".format(self.mode, n))

        self.transforms = all_transforms[self.mode]

    def __getitem__(self, index):

        rgb_path = self.rgb_paths[index]
        depth_path = self.depth_paths[index]
        gt_depth_path = self.gt_depth_paths[index]
        albedo_path = self.albedo_paths[index]

        img = Image.open(rgb_path)
        depth = Image.open(depth_path)
        gt_depth = Image.open(gt_depth_path)
        albedo = Image.open(albedo_path)
        # albedo = ImageOps.grayscale(albedo)

        # depth_array = np.asarray(gt_depth).astype(float)
        # depth_array = cv2.GaussianBlur(depth_array, (7, 7), 0)
        # depth = Image.fromarray(depth_array)

        sample = {'rgb': img, 'depth': depth, 'gt_depth': gt_depth, 'albedo': albedo}

        ret = self.transforms(sample)
        mask = torch.ones_like(ret['gt_depth'])
        mask[ret['gt_depth'] == -1] = 0
        mask[ret['depth'] == -1] = 0
        ret['mask'] = mask
        return ret

    def __len__(self):
        return len(self.rgb_paths)

