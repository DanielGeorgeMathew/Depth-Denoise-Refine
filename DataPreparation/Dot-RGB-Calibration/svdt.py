#!/usr/bin/env python

"""Calculates the transformation between two coordinate systems using SVD."""


import numpy as np


def svdt(A, B, order='col'):
    A, B = np.asarray(A), np.asarray(B)
    #print len(A),len(B)
    if order == 'row' or B.ndim == 1:
        if B.ndim == 1:
            A = A.reshape(A.size/3, 3)
            B = B.reshape(B.size/3, 3)
        R, L, RMSE = _svd(A, B)
    else:
        A = A.reshape(A.size/3, 3)
        ni = B.shape[0]
        R = np.empty((ni, 3, 3))
        L = np.empty((ni, 3))
        RMSE = np.empty(ni)
        for i in range(ni):
            R[i, :, :], L[i, :], RMSE[i] = _svd(A, B[i, :].reshape(A.shape))

    return R, L, RMSE


def _svd(A, B):
    Am = np.mean(A, axis=0)           # centroid of m1
    Bm = np.mean(B, axis=0)           # centroid of m2
    M = np.dot((B - Bm).T, (A - Am))  # considering only rotation
    # singular value decomposition
    U, S, Vt = np.linalg.svd(M)
    # rotation matrix
    R = np.dot(U, np.dot(np.diag([1, 1, np.linalg.det(np.dot(U, Vt))]), Vt))
    # translation vector
    L = B.mean(0)  - np.dot(R, A.mean(0))
    # RMSE
    err = 0
    for i in range(A.shape[0]):
        Bp = np.dot(R, A[i, :]) + L
        pre_err=err
        err += np.sum((Bp - B[i, :])**2)
        #print np.sqrt(err-pre_err)
    #print err
    
    RMSE = np.sqrt(err/A.shape[0]/3)
    #print RMSE
    return R, L, RMSE