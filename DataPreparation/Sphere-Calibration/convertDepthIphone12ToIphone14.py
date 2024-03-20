import numpy as np
import os
import cv2
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import json

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
    x = -x_over_z * z
    y = y_over_z * z
    return x, y, z


def visualizeDepthMap3D(depth_img, intrinsic):
    h, w = depth_img.shape
    pts_i12_org = []
    for u in range(w):
        for v in range(h):
            if abs(depth_img[v][u]) < 100 or depth_img[v][u] == 65535:
                # print('Invalid Depth at u,v: ', u, v)
                continue
            x, y, z = convert_from_uvd(u, v, depth_img[v][u], intrinsic)
            pts_i12_org.append([x, y, z])

    np.savetxt("/home/xyken/Downloads/sphereCalibrationData/iphone12/iphone12_original_RGBDCloud.txt",
               np.asarray(pts_i12_org))


def depth_convert(depth_img, intrinsic_12, intrinsic_14, projection_matrix, ext_12, ext_14):
    h, w = depth_img.shape
    gt_depth = np.zeros_like(depth_img)
    count = 0

    pts_i14 = []

    # visualizeDepthMap3D(depth_img, intrinsic_12)
    for u in range(w):
        for v in range(h):
            if abs(depth_img[v][u]) < 1 or depth_img[v][u] == 65535/30.0:
                # print('Invalid Depth at: ', u, v)
                continue
            x, y, z = convert_from_uvd(u, v, depth_img[v][u], intrinsic_12)

            hom_xyz = np.expand_dims(cv2.convertPointsToHomogeneous(np.asarray([[x, y, z]])).squeeze(), 1)

            hom_xyz = ext_12 @ hom_xyz
            print(hom_xyz)
            exit()
            proj_xyz = (projection_matrix @ hom_xyz)

            proj_xyz = (np.linalg.inv(ext_14) @ proj_xyz).squeeze()[:3]

            pts_i14.append(proj_xyz)

            proj_xyz_cv = np.copy(proj_xyz)
            proj_xyz_cv[1:3] = -proj_xyz_cv[1:3]
            # print(proj_xyz_cv)
            # print(proj_xyz_cv.dtype)
            # exit()
            uu, vv = cv2.projectPoints(proj_xyz_cv, (0, 0, 0), (0, 0, 0), intrinsic_14, None)[0].squeeze()

            comb = np.asarray(
                [[math.floor(uu), math.floor(vv)], [math.floor(uu), math.ceil(vv)], [math.ceil(uu), math.ceil(vv)],
                 [math.ceil(uu), math.floor(vv)]]).astype(int)

            for temp_u, temp_v in comb:
                if 0 <= temp_u < 480 and 0 <= temp_v < 640:
                    if gt_depth[temp_v][temp_u] == 0:
                        gt_depth[temp_v][temp_u] = -proj_xyz[2]
                    else:
                        gt_depth[temp_v][temp_u] = min(gt_depth[temp_v][temp_u], -proj_xyz[2])

    np.savetxt("/home/xyken/Daniel/SphereCalibrationMethods/Iphone-14/iphone14_from_iphone12_RGBDCloud.txt",
               np.array(pts_i14))

    return gt_depth


if __name__ == '__main__':
    root_path = '/home/xyken/Downloads/Daniel-Depth-Correction-Data/Daniel-Depth-Correction-Data'
    iphone12_root = os.path.join(root_path, 'iphone12/test-6.xcappdata/AppData/tmp')
    iphone14_root = os.path.join(root_path, 'iphone14/test-6.xcappdata/AppData/tmp')
    iphone12_depth_root = os.path.join(iphone12_root, 'depth')
    iphone14_depth_root = os.path.join(iphone14_root, 'depth')
    metadata_12_path = os.path.join(iphone12_root, 'metadata.json')
    metadata_14_path = os.path.join(iphone14_root, 'metadata.json')
    gt_depth_root = os.path.join(iphone14_root, 'gt_depth')
    if not os.path.exists(gt_depth_root):
        os.makedirs(gt_depth_root)
    extrinsics_12, intrinsic_12 = getExtrinsicIntrinsicFromMetadata(metadata_12_path)
    extrinsics_14, intrinsic_14 = getExtrinsicIntrinsicFromMetadata(metadata_14_path)

    transform = np.load(os.path.join(iphone12_root, 'transform12to14_sphereCalibration.npy'))

    for index, depth_path_12 in enumerate(sorted(glob(os.path.join(iphone12_depth_root, '*.png')),
                                                 key=lambda x: int(os.path.basename(x).split('.')[0]))[::20]):
        index = int(os.path.basename(depth_path_12).split('.')[0])
        depth_img_12 = cv2.imread(depth_path_12, cv2.IMREAD_ANYDEPTH).astype(float)
        # print(depth_img_12.shape)
        # print(depth_img_12.dtype)
        # print(np.unique(depth_img_12))
        # exit()
        depth_img_12 = depth_img_12 / 30.0

        gt_depth = (depth_convert(depth_img_12, intrinsic_12, intrinsic_14, transform, extrinsics_12[index].T,
                                  extrinsics_14[index].T)).astype(
            np.uint16) * 30

        cv2.imwrite(os.path.join(gt_depth_root, os.path.basename(depth_path_12)), gt_depth)
        index += 1
