import torch
from torch import nn
import matplotlib.pyplot as plt
import cv2
import numpy as np
import kornia
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from torchgeometry.losses import ssim, inverse_depth_smoothness_loss
from torchvision import transforms


def getDepthReal(depth_in):
    low_thresh = 0
    up_thresh = 65535
    thresh_range = (up_thresh - low_thresh) / 2.0
    ret = ((depth_in + 1) * thresh_range) / 30000.0
    return ret


def sobelLoss(gt, pred, mask, orig, beta=0.99999):
    # getEdges = Sobel(in_channels=1).cuda()
    # _, edges = kornia.filters.canny(orig.unsqueeze(1))
    # edges = kornia.morphology.dilation(edges, torch.ones((5,5)).cuda())
    _, depth_edges_orig = kornia.filters.canny(orig.unsqueeze(1))
    # _, depth_edges_14 = kornia.filters.canny(orig.unsqueeze(1))
    depth_edges_orig = kornia.morphology.dilation(depth_edges_orig, torch.ones((5, 5)).cuda())
    # depth_edges_14 = kornia.morphology.dilation(depth_edges_14, torch.ones((5, 5)).cuda())
    edges = torch.zeros_like(depth_edges_orig)
    edges[torch.where(depth_edges_orig == 1)] = 1
    # depth_edge_mask_12 = depth_edges_12.squeeze().cpu().numpy()
    # depth_edge_mask_14 = depth_edges_14.squeeze().cpu().numpy()

    diff = torch.abs(gt - pred)
    edge_mask = edges.squeeze()

    # fig, ax = plt.subplots(1, 4)
    # ax[0].imshow(edge_mask[0].cpu().numpy())
    # ax[1].imshow((edge_mask[0]*mask[0]).cpu().numpy())
    # ax[2].imshow(orig[0].cpu().numpy())
    # ax[3].imshow(mask[0].cpu().numpy())
    # plt.show()
    # exit()
    losses = torch.sum((beta * diff * edge_mask * mask + (1 - beta) * diff * (1 - edge_mask) * mask), dim=[1, 2])
    for i in range(gt.shape[0]):
        losses[i] = losses[i] / (torch.count_nonzero(mask[i]*edge_mask[i]) + 1)
    return torch.mean(losses)


def fidelity_loss(depth_pred, depth_gt, mask, L1=0.0, L2=1.0):
    ret = 0
    if L1 > 0.0:
        ret += L1 * nn.functional.l1_loss(depth_pred * mask, depth_gt * mask)
    if L2 > 0.0:
        ret += L2 * nn.functional.mse_loss(depth_pred * mask, depth_gt * mask)

    return ret


def depthLoss(gt, pred, mask):  # BxHxW
    _, h, w = gt.shape
    num_val = torch.count_nonzero(mask)
    # print(torch.clamp(torch.sum(torch.abs(gt - pred)*mask), max=300000))
    # print(num_val)
    # gt = gt * 30000 / 65535
    # mask = mask
    # pred = pred * 30000 / 65535
    # ssim_loss = ssim((gt * mask).unsqueeze(1), (pred * mask).unsqueeze(1), window_size=11, reduction='mean')
    # return ssim_loss
    # # print(ssim_loss)
    # # print(gt.shape)
    # # print(pred.shape)

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(gt[0].squeeze().detach().cpu().numpy())
    ax[1].imshow(pred[0].squeeze().detach().cpu().numpy())
    ax[2].imshow(mask[0].squeeze().detach().cpu().numpy())
    # ax[3].imshow(orig[0].squeeze().detach().cpu().numpy())
    plt.show()
    print(num_val)
    exit()
    # print('NUM VALID PIXELS : {}'.format(num_val))
    # print('TOTAL DEPTH LOSS : {}'.format(torch.clamp(torch.sum(torch.abs(gt - pred) * mask), max=300000)))
    diff = gt - pred
    l1 = torch.abs(diff)
    l1 = torch.where(mask == 0, mask, l1)
    l1 = torch.mean(l1)
    l2 = torch.square(diff)
    l2 = torch.where(mask == 0, mask, l2)
    l2 = torch.mean(l2)

    return l1 + l2
    # return torch.clamp(torch.sum(torch.abs(gt - pred) * mask), max=300000) / (num_val + 1)


def focalLoss(gt, pred, mask, alpha=0.25, gamma=2):
    _, h, w = gt.shape
    similarity = (1 - (torch.abs(gt - pred) * 30000 / 65535))
    losses = torch.sum(
        -alpha * torch.pow(1 - similarity, gamma) * torch.log(torch.clamp(similarity, min=0.01)) * mask, dim=[1, 2])
    for i in range(gt.shape[0]):
        losses[i] = losses[i] / (torch.count_nonzero(mask[i]) + 1)
    return torch.mean(losses)


def normaldotLoss(gt, pred, gt_normals, A, mask):
    fx = A[0][0]
    fy = A[1][1]
    cx = A[0][2]
    cy = A[1][2]
    inv_fx = 1.0 / fx
    inv_fy = 1.0 / fy
    b, h, w = gt.shape
    # gt_normals = depthToNormals(gt, A)  # BxHxWx3
    v_coord, u_coord = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    U_coord = torch.tile(u_coord.unsqueeze(0), [b, 1, 1]).cuda().float()
    V_coord = torch.tile(v_coord.unsqueeze(0), [b, 1, 1]).cuda().float()

    X_coord = -(U_coord - cx) * inv_fx * pred
    Y_coord = -(V_coord - cy) * inv_fy * pred
    Z_coord = pred

    nx = gt_normals[:, :, :, 0]
    ny = gt_normals[:, :, :, 1]
    nz = gt_normals[:, :, :, 2]

    v_coord, u_coord = torch.meshgrid(torch.arange(0, h))

    

    product = torch.square((X_coord[:, :-2, 1:-1] - X_coord[:, 1:-1, 1:-1]) * nx[:, 1:-1, 1:-1] +
                           (Y_coord[:, :-2, 1:-1] - Y_coord[:, 1:-1, 1:-1]) * ny[:, 1:-1, 1:-1] +
                           (Z_coord[:, :-2, 1:-1] - Z_coord[:, 1:-1, 1:-1]) * nz[:, 1:-1, 1:-1]) + \
              torch.square((X_coord[:, 2:, 1:-1] - X_coord[:, 1:-1, 1:-1]) * nx[:, 1:-1, 1:-1] +
                           (Y_coord[:, 2:, 1:-1] - Y_coord[:, 1:-1, 1:-1]) * ny[:, 1:-1, 1:-1] +
                           (Z_coord[:, 2:, 1:-1] - Z_coord[:, 1:-1, 1:-1]) * nz[:, 1:-1, 1:-1]) + \
              torch.square((X_coord[:, 1:-1, :-2] - X_coord[:, 1:-1, 1:-1]) * nx[:, 1:-1, 1:-1] +
                           (Y_coord[:, 1:-1, :-2] - Y_coord[:, 1:-1, 1:-1]) * ny[:, 1:-1, 1:-1] +
                           (Z_coord[:, 1:-1, :-2] - Z_coord[:, 1:-1, 1:-1]) * nz[:, 1:-1, 1:-1]) + \
              torch.square((X_coord[:, 1:-1, 2:] - X_coord[:, 1:-1, 1:-1]) * nx[:, 1:-1, 1:-1] +
                           (Y_coord[:, 1:-1, 2:] - Y_coord[:, 1:-1, 1:-1]) * ny[:, 1:-1, 1:-1] +
                           (Z_coord[:, 1:-1, 2:] - Z_coord[:, 1:-1, 1:-1]) * nz[:, 1:-1, 1:-1])

    paddings = (1, 1, 1, 1)

    product = torch.nn.functional.pad(product, paddings, mode='constant')

    product = torch.where(mask == 0, mask, product)
    # product = torch.sum(product) / torch.count_nonzero(mask)
    product = torch.sum(product)/(2*b)
    return product


def smoothness_loss(depth_pred, mask):
    # diff_1 = torch.sum(torch.abs(depth_pred[:, :, 1:-1, 1:-1] - depth_pred[:, :, 0:-2, 1:-1]) * mask[:, :, 1:-1, 1:-1])
    # diff_2 = torch.sum(torch.abs(depth_pred[:, :, 1:-1, 1:-1] - depth_pred[:, :, 1:-1, 0:-2]) * mask[:, :, 1:-1, 1:-1])
    ret = kornia.losses.total_variation(depth_pred * mask, reduction='mean').mean()
    return ret


def DenoiseNetLoss(depth_gt, depth_pred, normals_gt, normals_pred, mask, A, orig):
    depth_gt = depth_gt.squeeze()
    depth_pred = depth_pred.squeeze()
    mask = mask.squeeze()
    normals_gt = torch.permute(normals_gt, [0, 2, 3, 1])
    orig = orig.squeeze()

    lambda_reconstruction = 1.0
    lambda_normaldot = 3000
    lambda_normalsmooth = 0.5
    lambda_edge = 0.0
    lambda_smooth = 0.0

    # recontructionLoss = depthLoss(depth_gt, depth_pred, mask)
    recontructionLoss = fidelity_loss(depth_gt, depth_pred, mask*65535/2.0, L1=1.0, L2=1.0)
    normalDotLoss = normaldotLoss(depth_gt, getDepthReal(depth_pred), normals_gt, A, mask)
    edgeLoss = sobelLoss(depth_gt, depth_pred, mask, orig)
    smoothLoss = smoothness_loss(depth_pred.unsqueeze(1), mask.unsqueeze(1))
    normalSmoothLoss = smoothness_loss(normals_pred, mask.unsqueeze(1))

    # print(recontructionLoss)
    # print(normalDotLoss)
    # print(edgeLoss)
    # print(smoothLoss)
    # print(normalSmoothLoss)
    # exit()
    ret = lambda_reconstruction * recontructionLoss + lambda_normaldot * normalDotLoss + lambda_edge * edgeLoss + lambda_smooth * smoothLoss \
          + lambda_normalsmooth * normalSmoothLoss
    return ret
