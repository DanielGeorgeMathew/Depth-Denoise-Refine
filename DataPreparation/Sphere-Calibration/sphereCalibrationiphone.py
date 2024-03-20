import numpy as np
import os
import cv2
import yaml
import torch
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import open3d as o3d
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


root_path = '/home/xyken/Downloads/Daniel-Depth-Correction-Data/Daniel-Depth-Correction-Data'
iphone12_root = os.path.join(root_path, 'iphone12/test-6.xcappdata/AppData/tmp')
iphone14_root = os.path.join(root_path, 'iphone14/test-6.xcappdata/AppData/tmp')

iphone12_spheres = []
iphone14_spheres = []
for i in range(1, 5):
    iphone12_spheres.append(o3d.io.read_point_cloud(os.path.join(iphone12_root, 'spheres/sphere_{}.ply'.format(i))))
    iphone14_spheres.append(o3d.io.read_point_cloud(os.path.join(iphone14_root, 'spheres/sphere_{}.ply'.format(i))))

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
    # o3d.visualization.draw_geometries([iphone12_spheres[i], bestfit_12_sphere])
    # o3d.visualization.draw_geometries([iphone14_spheres[i],
    #                                    bestfit_14_sphere])
    iphone12_centers.append(best_fit_12.point)
    iphone14_centers.append(best_fit_14.point)

T, RMSE = getTransform(np.asarray(iphone12_centers), np.asarray(iphone14_centers))

iphone12_mesh = o3d.io.read_triangle_mesh(os.path.join(iphone12_root, 'truedepth_mesh.ply'))
iphone14_mesh = o3d.io.read_triangle_mesh(os.path.join(iphone14_root, 'truedepth_mesh.ply'))

o3d.io.write_triangle_mesh(os.path.join(iphone12_root, 'iphone12_mesh_alligned_with_iphone14.ply'),
                           iphone12_mesh.transform(T))
np.save(os.path.join(iphone12_root, 'transform12to14_sphereCalibration.npy'), T)
print(T, RMSE)
