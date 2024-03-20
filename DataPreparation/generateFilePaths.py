import numpy as np
import os
import cv2
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import json
import random

REF_W, REF_H = 3024, 4032
def getExtrinsicIntrinsicFromMetadata(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata_dict = json.load(f)
    extrinsic_list = np.asarray([i['matrix'] for i in metadata_dict['extrinsic']])
    intrinsic = np.asarray(metadata_dict['intrinsic']['matrix']).T
    intrinsic[0][0] = intrinsic[0][0] * 240 / REF_W
    intrinsic[1][1] = intrinsic[1][1] * 320 / REF_H
    intrinsic[0][2] = intrinsic[0][2] * 240 / REF_W
    intrinsic[1][2] = intrinsic[1][2] * 320 / REF_H
    return extrinsic_list, intrinsic


def getRGBPath(x):
    file_name = os.path.basename(x)
    dir_name = os.path.dirname(x)
    dir_name = dir_name.replace('gt_depth', 'rgb')
    return os.path.join(dir_name, file_name)


def getDepthPath(x):
    file_name = os.path.basename(x)
    dir_name = os.path.dirname(x)
    dir_name = dir_name.replace('gt_depth', 'high_quality_depths')
    return os.path.join(dir_name, file_name)


def getAlbedoPath(x):
    file_name = os.path.basename(x)
    dir_name = os.path.dirname(x)
    dir_name = dir_name.replace('gt_depth', 'albedo')
    return os.path.join(dir_name, file_name)


target_path = '/home/xyken/Daniel/DepthRefinement/DataPreparation/DDRNet-OfficeData-v3'
if not os.path.exists(target_path):
    os.makedirs(target_path)
train_target_path = os.path.join(target_path, 'train.txt')
val_target_path = os.path.join(target_path, 'val.txt')
intrinsic_target_path = os.path.join(target_path, 'intrinsic.npy')
# train_ids = list(range(1, 11))
# del (train_ids[2])
# del (train_ids[2])
# train_ids = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13]
train_ids = range(1, 40)
lines = []
for id in train_ids:
    root_path = '/home/xyken/Downloads/DDRNet-TrainData-v2/DDRNet-TrainData-v2/scan-{}.xcappdata/AppData/tmp/'.format(id)
    gt_depth_paths = glob(os.path.join(root_path, 'gt_depth/*.png'))
    for i in gt_depth_paths:
        lines.append('{} {} {} {}\n'.format(getRGBPath(i), getDepthPath(i), getAlbedoPath(i), i))

random.shuffle(lines)

num_total = len(lines)
num_train = int(0.8 * num_total)

lines_train = lines[:num_train]
lines_val = lines[num_train:]

print(len(lines_train))
print(len(lines_val))

with open(train_target_path, 'w') as fp:
    fp.writelines(lines_train)

with open(val_target_path, 'w') as fp:
    fp.writelines(lines_val)


intrinsic_avg = np.zeros((3, 3))
for j in tqdm(train_ids):
    root_path = '/home/xyken/Downloads/DDRNet-TrainData-v2/DDRNet-TrainData-v2/scan-{}.xcappdata/AppData/tmp/'.format(j)
    _, intrinsic = getExtrinsicIntrinsicFromMetadata(os.path.join(root_path, 'metadata.json'))
    print("INDEX-{} Metadata".format(j))
    print(intrinsic)
    intrinsic_avg += intrinsic
print("INTRINSIC AVERAGE IS:")
print(intrinsic_avg/len(train_ids))
np.save(intrinsic_target_path, intrinsic_avg/len(train_ids))
