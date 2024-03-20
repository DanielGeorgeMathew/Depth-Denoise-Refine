import json
import numpy as np
import os
import cv2
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
plt.rcParams['image.cmap'] = 'gray'
imageSize = (3024, 4032)
stereo_calibration_root = '/home/xyken/Daniel/calibration/stereo_2/results'

iphone12_image = cv2.imread('/home/xyken/Daniel/calibration/stereo_2/iphone12_calibration_images/IMG_0473.jpg',0)
iphone14_image = cv2.imread('/home/xyken/Daniel/calibration/stereo_2/iphone14_calibration_images/IMG_6337.jpg',0)

fig, ax = plt.subplots(1,2)
ax[0].imshow(iphone12_image)
ax[1].imshow(iphone14_image)
plt.show()


R = np.load(os.path.join(stereo_calibration_root, 'rotation.npy'))
T = np.load(os.path.join(stereo_calibration_root, 'translation.npy'))

K_12 = np.load(os.path.join(stereo_calibration_root, 'intrinsic_12.npy'))
K_14 = np.load(os.path.join(stereo_calibration_root, 'intrinsic_14.npy'))

D_12 = np.load(os.path.join(stereo_calibration_root, 'dist_12.npy'))
D_14 = np.load(os.path.join(stereo_calibration_root, 'dist_14.npy'))

R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K_12, D_12, K_14, D_14, imageSize, R, T,
                                         None, None, None, None, None,
                                         cv2.CALIB_ZERO_DISPARITY
                                         )

leftmapX, leftmapY = cv2.initUndistortRectifyMap(
        K_12, D_12, R1,
        P1, imageSize, cv2.CV_32FC1)

rightmapX, rightmapY = cv2.initUndistortRectifyMap(
        K_14, D_14, R2,
        P2, imageSize, cv2.CV_32FC1)

print(leftmapX)
print(leftmapX.shape)
exit()
