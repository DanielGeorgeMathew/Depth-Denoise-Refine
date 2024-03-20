import numpy as np
import os
import cv2
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import json
import open3d as o3d


def depthToPointCloud(depth_img, rgb_img, target_path, intrinsic_14):
    pcd = o3d.geometry.PointCloud()
    xyz = []
    colors = []
    h, w = depth_img.shape

    for u in range(h):
        for v in range(w):
            xyz.append(convert_from_uvd(u, v, depth_img[u][v], intrinsic_14))
            colors.append(rgb_img[u][v])

    xyz = np.asarray(xyz)
    colors = np.asarray(colors) / 255.0

    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    print(np.unique(xyz))
    print(xyz)
    o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(target_path, pcd)


def convert_from_uvd(u, v, d, A):
    fx = A[0][0]
    fy = A[1][1]
    cx = A[0][2]
    cy = A[1][2]

    x_over_z = (u - cx) / fx
    y_over_z = (v - cy) / fy
    z = -d
    x = -x_over_z * z
    y = y_over_z * z
    return x, y, z


ref_w, ref_h = 3024, 4032
intrinsic_14 = np.zeros((3, 3))
intrinsic_14[0][0] = 2712.72265 * 480 / ref_w
intrinsic_14[1][1] = 2712.72265 * 640 / ref_h
intrinsic_14[0][2] = 1512 * 480 / ref_w
intrinsic_14[1][2] = 2015.9998 * 640 / ref_h

root_path = '/home/xyken/Downloads/sphereCalibrationData_1'
target_path = os.path.join(root_path, 'iphone14/cleaned_depth')
gt_depth_path = os.path.join(root_path, 'iphone14/gt_depth')
depth_path = os.path.join(root_path, 'iphone14/depth')
rgb_path = os.path.join(root_path, 'iphone14/rgb')

gt_depth_list = sorted(glob(os.path.join(gt_depth_path, '*.png')),
                       key=lambda x: int(os.path.basename(x).split('.')[0]))
depth_list = sorted(glob(os.path.join(depth_path, '*.png')),
                    key=lambda x: int(os.path.basename(x).split('.')[0]))
rgb_list = sorted(glob(os.path.join(rgb_path, '*.png')),
                  key=lambda x: int(os.path.basename(x).split('.')[0]))

for i, j, k in tqdm(zip(gt_depth_list, depth_list, rgb_list)):
    gt_depth = cv2.imread(i, cv2.IMREAD_ANYDEPTH).astype(float) / 30.0
    depth = cv2.imread(j, cv2.IMREAD_ANYDEPTH).astype(float) / 30.0
    rgb = cv2.cvtColor(cv2.imread(k), cv2.COLOR_BGR2RGB)

    gt_depth_cp = np.copy(gt_depth)
    gt_depth_cp[depth == 0] = 0
    depth_cp = np.copy(depth)
    depth_cp[gt_depth == 0] = 0

    diff = np.abs(gt_depth_cp - depth_cp)
    mask = np.masked_inside(np.abs(gt_depth - depth), 0, 35)
    cv2.imwrite(os.path.join(target_path, 'iphone12', os.path.basename(i).split('.')[0] + '.png'),
                (gt_depth * mask).astype(np.uint16) * 30)
    cv2.imwrite(os.path.join(target_path, 'iphone14', os.path.basename(i).split('.')[0] + '.png'),
                (depth * mask).astype(np.uint16) * 30)
    depthToPointCloud(depth_cp*mask, rgb, os.path.join(target_path, os.path.basename(i).split('.')[0] + '.ply'), intrinsic_14)
    exit()

    fig, ax = plt.subplots(1, 4)
    ax[0].imshow(gt_depth, cmap='gray')
    ax[0].set_title('iphone12')
    ax[1].imshow(depth, cmap='gray')
    ax[1].set_title('iphone14')
    ax[2].imshow(gt_depth_cp*mask, cmap='gray')
    ax[2].set_title('masked_12')
    ax[3].imshow(depth_cp*mask, cmap='gray')
    ax[3].set_title('masked_14')
    plt.show()

