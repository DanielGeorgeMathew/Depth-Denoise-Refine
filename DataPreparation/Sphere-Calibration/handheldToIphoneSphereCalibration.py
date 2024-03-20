import numpy as np
import open3d as o3d
import cv2
import os
import yaml
import torch
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from skspatial.objects import Sphere
import svdt


def getTransform(points_source, points_target, ):
    try:
        T = np.eye(4)
        R, L, RMSE = svdt.svdt(points_source, points_target, order='row')
        T[0:3, 0:3] = R
        T[0:3, 3] = L
        return T, RMSE
    except:
        pass


def readEinscanPLY(mesh_path):
    with open(mesh_path, 'r') as f:
        lines = f.readlines()
    points = [i.rstrip().split(' ') for i in lines[9:]]
    points = np.float32([i for i in points if len(i) == 3])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def mirrorPointCloudXZ(pcd):
    pcd_mirrored = o3d.geometry.PointCloud(pcd)
    temp = np.asarray(pcd_mirrored.points)
    temp[:, 2] = -temp[:, 2]
    # temp[:, 0] = -temp[:, 0]
    pcd_mirrored.points = o3d.utility.Vector3dVector(temp)
    return pcd_mirrored


root_path = '/home/xyken/Downloads/final-Training-Data'

iphone14_root = os.path.join(root_path, 'iphone-14/spheres')
gt_root = os.path.join(root_path, 'handheld/spheres')

iphone12_spheres = []
iphone14_spheres = []
for i in range(1, 5):
    gt_sphere = o3d.io.read_point_cloud(os.path.join(gt_root, 'sphere_{}.ply'.format(i)))
    iphone14_sphere = o3d.io.read_point_cloud(os.path.join(iphone14_root, 'sphere_{}.ply'.format(i)))
    iphone12_spheres.append(gt_sphere)
    iphone14_spheres.append(iphone14_sphere)

# o3d.visualization.draw_geometries(iphone14_spheres + iphone12_spheres)

iphone12_centers = []
iphone14_centers = []

for i in range(4):
    sphere_points_12 = np.asarray(iphone12_spheres[i].points)
    sphere_points_14 = np.asarray(iphone14_spheres[i].points)
    print(sphere_points_12.shape)
    print(sphere_points_14.shape)

    best_fit_12 = Sphere.best_fit(sphere_points_12)
    best_fit_14 = Sphere.best_fit(sphere_points_14)

    bestfit_12_sphere = o3d.geometry.TriangleMesh().create_sphere(radius=best_fit_12.radius).translate(
        best_fit_12.point, relative=False)
    bestfit_14_sphere = o3d.geometry.TriangleMesh().create_sphere(radius=best_fit_14.radius).translate(
        best_fit_14.point, relative=False)
    iphone14_spheres[i].paint_uniform_color([1, 0, 0])
    iphone12_spheres[i].paint_uniform_color([1, 0, 0])
    print(best_fit_12.radius)
    print(best_fit_14.radius)
    # o3d.visualization.draw_geometries([iphone12_spheres[i], bestfit_12_sphere, iphone14_spheres[i],
    #                                    bestfit_14_sphere])
    # o3d.visualization.draw_geometries([iphone14_spheres[i],
    #                                    bestfit_14_sphere])
    iphone12_centers.append(best_fit_12.point)
    iphone14_centers.append(best_fit_14.point)

T, RMSE = getTransform(np.asarray(iphone12_centers), np.asarray(iphone14_centers))

# iphone12_mesh = readEinscanPLY(os.path.join(root_path, 'handheld/transformed-final-handheld-scan.ply'))
iphone12_mesh = o3d.io.read_point_cloud(os.path.join(root_path, 'handheld/transformed-final-handheld-scan.ply'))
iphone14_mesh = o3d.io.read_triangle_mesh(os.path.join(root_path, 'iphone-14/truedepth_mesh.ply'))

# o3d.visualization.draw_geometries([iphone12_mesh, iphone14_mesh, iphone12_mesh_mirrored])
# exit()
o3d.io.write_point_cloud(os.path.join(root_path, 'handheld/mesh_transformed.ply'),
                         iphone12_mesh.transform(T))

np.save(os.path.join(root_path, 'handheld/transform12to14_sphereCalib.npy'), T)
print(T, RMSE)
