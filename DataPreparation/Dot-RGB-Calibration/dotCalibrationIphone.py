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

root_path = '~/Downloads/compareCalibrationData'
iphone12_root = os.path.join(root_path, 'iphone-12')
iphone14_root = os.path.join(root_path, 'iphone-14')


def detectBlobs(img):
    params = cv2.SimpleBlobDetector_Params()

    params.blobColor = 0
    params.minThreshold = 20
    # params.maxThreshold = 255
    params.filterByArea = True
    # params.minArea = 1500
    params.maxArea = 250
    params.filterByCircularity = True
    params.minCircularity = 0.8
    params.filterByConvexity = True
    params.minConvexity = 0.8
    params.filterByInertia = True
    params.minInertiaRatio = 0.8

    detector = cv2.SimpleBlobDetector_create(params)
    # cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cimg = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img,(5,5),0)

    keypoints = detector.detect(img)
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # circles = np.uint16(np.around(circles))
    # for i in circles[0, :]:
    #     # draw the outer circle
    #     cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    #     # draw the center of the circle
    #     cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)
    #
    cv2.imshow('detected circles', im_with_keypoints)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return keypoints


def detectCircles(img):
    # cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cimg = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.medianBlur(img, 5)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20,
                              param1=30, param2=50, minRadius=5, maxRadius=10)
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('detected circles', cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return circles

def getFirstFrame(root_path):
    vidcap = cv2.VideoCapture(os.path.join(root_path, 'video.mp4'))
    success, image = vidcap.read()
    if success:
        cv2.imwrite(os.path.join(root_path,"first_frame.jpg"), image)
        return image


def getTransform(points_source, points_target, ):
    try:
        T = np.eye(4)
        R, L, RMSE = svdt.svdt(points_source, points_target, order='row')
        T[0:3, 0:3] = R
        T[0:3, 3] = L
        return T, RMSE
    except:
        pass


iphone12_img = getFirstFrame(iphone12_root)
iphone14_img = getFirstFrame(iphone14_root)

circles_12 = detectBlobs(iphone12_img)
circles_14 = detectBlobs(iphone14_img)
# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(iphone12_img)
# ax[0].set_title('iphone12')
# ax[1].imshow(iphone14_img)
# ax[1].set_title('iphone14')
# plt.show()


