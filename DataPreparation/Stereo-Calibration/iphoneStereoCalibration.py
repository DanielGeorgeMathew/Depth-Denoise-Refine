import json
import numpy as np
import os
import cv2 as cv
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import math


def stereo_calibrate(frames_folder, cm_init_1, cm_init_2):
    # read the synched frames
    c1_images_names = sorted(glob(os.path.join(frames_folder, 'iphone12_calibration_images/*.jpg')))
    c2_images_names = sorted(glob(os.path.join(frames_folder, 'iphone14_calibration_images/*.jpg')))

    c1_images = []
    c2_images = []
    for im1, im2 in zip(c1_images_names, c2_images_names):
        _im = cv.imread(im1, 1)
        c1_images.append(_im)

        _im = cv.imread(im2, 1)
        c2_images.append(_im)

    # change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    rows = 17  # number of checkerboard rows.
    columns = 27  # number of checkerboard columns.
    world_scaling = 5.  # change this to the real world square size. Or not.

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows * columns, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:columns].T.reshape(-1, 2)
    objp = world_scaling * objp

    # frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]

    # Pixel coordinates of checkerboards
    imgpoints_left = []  # 2d points in image plane.
    imgpoints_right = []

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space
    num_pairs = 0
    for frame1, frame2 in tqdm(list(zip(c1_images, c2_images))):
        gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
        # c_ret1, corners1 = cv.findChessboardCorners(gray1, (5, 8), None)
        # c_ret2, corners2 = cv.findChessboardCorners(gray2, (5, 8), None)
        gray1 = cv.blur(gray1, (5,5), cv.BORDER_DEFAULT)
        gray2 = cv.blur(gray2, (5,5), cv.BORDER_DEFAULT)
        c_ret1, corners1 = cv.findChessboardCornersSB(gray1, (rows, columns),
                                                      flags=cv.CALIB_CB_EXHAUSTIVE + cv.CALIB_CB_ACCURACY + cv.CALIB_CB_LARGER)
        c_ret2, corners2 = cv.findChessboardCornersSB(gray2, (rows, columns),
                                                      flags=cv.CALIB_CB_EXHAUSTIVE + cv.CALIB_CB_ACCURACY + cv.CALIB_CB_LARGER)
        # print(c_ret1, c_ret2)
        # continue
        if c_ret1 == True and c_ret2 == True:
            num_pairs += 1
            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
            print(corners1)
            print('*******************')
            print('*******************')
            print('*******************')
            print('*******************')
            print('*******************')


            print(corners2)
            print('*******************')
            print('*******************')
            print('*******************')
            print('*******************')
            print('*******************')
            frame_old_1 = np.copy(frame1)
            frame_old_2 = np.copy(frame2)
            cv.drawChessboardCorners(frame_old_1, (rows, columns), corners1, c_ret1)
            # cv.imshow('img', frame1)

            cv.drawChessboardCorners(frame_old_2, (rows, columns), corners2, c_ret2)
            # cv.imshow('img2', frame2)
            # k = cv.waitKey(500)
            fig, ax = plt.subplots(1, 2, figsize=[50, 50])

            ax[0].imshow(cv.cvtColor(frame_old_1, cv.COLOR_RGB2BGR))
            ax[1].imshow(cv.cvtColor(frame_old_2, cv.COLOR_RGB2BGR))

            plt.show()
            # exit()

            choice = input('Enter which pattern to reverse left(1) or right(2) or 0 for neither!!')
            if choice == '1':
                corners1 = corners1[::-1]
                print(corners1)
            elif choice == '2':
                corners2 = corners2[::-1]
                print(corners2)
            elif choice == '3':
                corners1 = corners1[::-1]
                corners2 = corners2[::-1]
                print(corners1)
                print('****')
                print(corners2)

            if choice != '0':
                frame_new_1 = np.copy(frame1)
                frame_new_2 = np.copy(frame2)
                cv.drawChessboardCorners(frame_new_1, (rows, columns), corners1, c_ret1)
                # cv.imshow('img', frame1)

                cv.drawChessboardCorners(frame_new_2, (rows, columns), corners2, c_ret2)
                # cv.imshow('img2', frame2)
                # k = cv.waitKey(500)
                fig, ax = plt.subplots(1, 2, figsize=[50, 50])

                ax[0].imshow(cv.cvtColor(frame_new_1, cv.COLOR_RGB2BGR))
                ax[1].imshow(cv.cvtColor(frame_new_2, cv.COLOR_RGB2BGR))

                plt.show()

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
    print('{} pairs used for stereo calibration'.format(num_pairs))
    stereocalibration_flags = cv.CALIB_FIX_FOCAL_LENGTH + cv.CALIB_FIX_PRINCIPAL_POINT

    # cm_init_1 = np.identity(3)
    # cm_init_2 = np.identity(3)
    dist_init_1 = np.zeros(5)
    dist_init_2 = np.zeros(5)
    # print(cm_init_1)
    # print(cm_init_2)
    # exit()

    ret, CM1, dist1, CM2, dist2, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, np.ascontiguousarray(cm_init_1),
                                                                 np.ascontiguousarray(cm_init_2), dist_init_1, dist_init_2,
                                                                 imageSize=(3088, 2316), criteria=criteria, flags = 0
                                                                 )

    print('Stereo calibration error is {}'.format(ret))

    return CM1, CM2, dist1, dist2, R, T


if __name__ == '__main__':
    iphone12_root = '/home/xyken/Daniel/calibration/iphone_12'
    iphone14_root = '/home/xyken/Daniel/calibration/iphone_14'
    stereo_calibration_root = '/home/xyken/Daniel/calibration/stereo_2'
    intrinsic_12_path = '/home/xyken/Daniel/calibration/iphone12_depth_stream_2/metadata.json'
    intrinsic_14_path = '/home/xyken/Daniel/calibration/iphone14_depth_stream_2/metadata.json'

    with open(intrinsic_12_path, 'r') as f:
        int_12_dict = json.load(f)
    intrinsic_12 = np.transpose(np.asarray(int_12_dict['intrinsic']['matrix']))
    with open(intrinsic_14_path, 'r') as f:
        int_14_dict = json.load(f)
    intrinsic_14 = np.transpose(np.asarray(int_14_dict['intrinsic']['matrix']))

    # intrinsic_12[0][0] = intrinsic_12[0][0] * 2316 / 3024
    # intrinsic_12[1][1] = intrinsic_12[1][1] * 3088 / 4032
    # intrinsic_12[0][2] = intrinsic_12[0][2] * 2316 / 3024
    # intrinsic_12[1][2] = intrinsic_12[1][2] * 3088 / 4032
    #
    # intrinsic_14[0][0] = intrinsic_14[0][0] * 2316 / 3024
    # intrinsic_14[1][1] = intrinsic_14[1][1] * 3088 / 4032
    # intrinsic_14[0][2] = intrinsic_14[0][2] * 2316 / 3024
    # intrinsic_14[1][2] = intrinsic_14[1][2] * 3088 / 4032

    # print(np.identity(3))
    # print(intrinsic_12)
    # exit()
    # mtx1 = np.load(os.path.join(iphone12_root, 'intrinsic_2.npy'))
    # dist1 = np.load(os.path.join(iphone12_root, 'distortion_coeff_2.npy'))
    # mtx2 = np.load(os.path.join(iphone14_root, 'intrinsic_2.npy'))
    # dist2 = np.load(os.path.join(iphone14_root, 'distortion_coeff_2.npy'))

    mtx1, mtx2, dist1, dist2, R, T = stereo_calibrate(stereo_calibration_root, intrinsic_12, intrinsic_14)

    print('Rotation from iphone 12 to iphone 14 is {}\n'.format(R))

    print('Translation from iphone 12 to iphone 14 is {}\n'.format(T))

    print('Camera matrix for iphone12 is {}\n'.format(mtx1))
    print('Camera matrix for iphone14 is {}\n'.format(mtx2))

    print('Distortion Coefficiants for iphone12 is {}\n'.format(dist1))
    print('Distortion Coefficiants for iphone14 is {}\n'.format(dist2))

    np.save(os.path.join(stereo_calibration_root, 'results/rotation.npy'), np.asarray(R), allow_pickle=True)
    np.save(os.path.join(stereo_calibration_root, 'results/translation.npy'), np.asarray(T), allow_pickle=True)

    np.save(os.path.join(stereo_calibration_root, 'results/intrinsic_12.npy'), np.asarray(mtx1), allow_pickle=True)
    np.save(os.path.join(stereo_calibration_root, 'results/intrinsic_14.npy'), np.asarray(mtx2), allow_pickle=True)

    np.save(os.path.join(stereo_calibration_root, 'results/dist_12.npy'), np.asarray(dist1), allow_pickle=True)
    np.save(os.path.join(stereo_calibration_root, 'results/dist_14.npy'), np.asarray(dist2), allow_pickle=True)
