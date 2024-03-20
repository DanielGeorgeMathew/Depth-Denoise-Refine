import torch
import torchvision.transforms.functional
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


def H0(normals):
    return torch.ones_like(normals, dtype=torch.float32)[:, 0:1, :]


def H1(normals):
    return normals[:, 1:2, :]


def H2(normals):
    return normals[:, 2:3, :]


def H3(normals):
    return normals[:, 0:1, :]


def H4(normals):
    return normals[:, 0:1, :] * normals[:, 1:2, :]


def H5(normals):
    return normals[:, 1:2, :] * normals[:, 2:3, :]


def H6(normals):
    return -(normals[:, 0:1, :] * normals[:, 0:1, :]) \
        - (normals[:, 1:2, :] * normals[:, 1:2, :]) \
        + 2 * (normals[:, 2:3, :] * normals[:, 2:3, :])


def H7(normals):
    return normals[:, 2:3, :] * normals[:, 0:1, :]


def H8(normals):
    return (normals[:, 0:1, :] * normals[:, 0:1, :]) \
        - (normals[:, 1:2, :] * normals[:, 1:2, :])


def get_spherical_harmonics(normals, mask):
    b, c, h, w = normals.shape
    normals = torch.reshape(normals, [-1, c, h * w])
    mask = torch.reshape(mask, [-1, 1, h * w])
    return torch.cat([H0(normals), H1(normals), H2(normals), H3(normals), H4(normals),
                      H5(normals), H6(normals), H7(normals), H8(normals)], dim=1) * mask


def get_lighting_coefficiants(normals, image, abd, mask, rm_graz=False, eps=1e-4):
    b, c, h, w = image.shape
    if rm_graz:
        mask_grazed = torch.abs(normals[:, 2:3, :, :]) > 0.3
        mask = mask * mask_grazed
    # fig, ax = plt.subplots(1, 4)
    # ax[0].imshow(mask.squeeze()[0].cpu().detach().numpy())
    # ax[1].imshow(mask_grazed.squeeze()[0].cpu().detach().numpy())
    # ax[2].imshow(image.squeeze()[0].cpu().detach().numpy())
    # ax[3].imshow(normals[0, 2, :, :].cpu().detach().numpy())
    # plt.show()
    # exit()
    A = get_spherical_harmonics(normals, mask)  # B x 9 x H*W
    # print(abd.shape)
    # exit()
    coeff_list = []
    for channel in range(c):
        curr_img = torch.permute(torch.reshape(image[:, channel, :, :].unsqueeze(1), [-1, 1, h * w]), [0, 2, 1])
        if abd is not None:
            curr_abd = torch.reshape(abd[:, channel, :, :].unsqueeze(1), [-1, 1, h * w])
            curr_A = A * curr_abd
        else:
            curr_A = A
        curr_A_t = torch.permute(curr_A, dims=[0, 2, 1])  # B x H*W x 9
        curr_AA_t = torch.matmul(curr_A, curr_A_t) + eps * torch.eye(9).cuda()  # B x 9 x 9

        lighting_coeffs = torch.matmul(torch.matmul(torch.linalg.inv(curr_AA_t), curr_A), curr_img).squeeze(2)
        coeff_list.append(lighting_coeffs.unsqueeze(1))

    lighting_coeffs = torch.concat(coeff_list, dim=1)
    LH = torch.reshape(torch.matmul(lighting_coeffs, A), [-1, c, h, w])
    return lighting_coeffs, LH, mask


def shading_loss(normals_pred, normals_gt, color, mask, albedo=None, grad_metric=False, avg_pool=False,
                 grad_weight=1e4, is_color=False):

    albedo = mask * torch.ones_like(color)
    if not is_color:
        color = transforms.functional.rgb_to_grayscale(color)
    if avg_pool:
        color = nn.functional.avg_pool2d(color, kernel_size=3, stride=1, padding=1, count_include_pad=False)

    b, c, h, w = color.shape
    color = (color + 1.0) / 2.0

    # color = torchvision.transforms.functional.adjust_contrast(color[:, :, :, :], contrast_factor=1.5)

    l_coefs, LH, _ = get_lighting_coefficiants(normals_gt, color, albedo, mask, rm_graz=True)
    A = get_spherical_harmonics(normals_pred, mask)  # B x 9 x H*W
    irradiance = color * torch.matmul(l_coefs, A).reshape([-1, c, h, w])
    irradiance = torch.clamp(irradiance, min=0, max=1)

    # albedo_est = torch.divide(LH, gray)
    # irradiance_est = albedo_est * torch.matmul(l_coefs.unsqueeze(1), A).reshape([-1, 1, h, w])
    # irradiance_est = torch.clamp(irradiance_est, min=0, max=1)
    # fig, ax = plt.subplots(1, 4)
    # ax[0].imshow(torch.permute(irradiance, dims=[0, 2, 3, 1]).squeeze()[0].cpu().detach().numpy())
    # ax[1].imshow(torch.permute(color, dims=[0, 2, 3, 1]).squeeze()[0].cpu().detach().numpy())
    # ax[2].imshow(torch.permute(albedo, dims=[0, 2, 3, 1]).squeeze()[0].cpu().detach().numpy())
    # ax[3].imshow(mask.squeeze()[0].cpu().detach().numpy())
    # plt.show()
    # exit()
    # irradiance = irradiance_est
    grad_loss = 0
    if grad_metric:
        color_grad_y = color[:, :, 1:-1, 1:-1] - color[:, :, :-2, 1:-1]
        color_grad_x = color[:, :, 1:-1, 1:-1] - color[:, :, 1:-1, :-2]
        irradiance_grad_y = irradiance[:, :, 1:-1, 1:-1] - irradiance[:, :, :-2, 1:-1]
        irradiance_grad_x = irradiance[:, :, 1:-1, 1:-1] - irradiance[:, :, 1:-1, :-2]

        weights = mask[:, :, 1:-1, 1:-1]
        grad_loss = nn.functional.mse_loss(irradiance_grad_y * weights, color_grad_y * weights) + \
                    nn.functional.mse_loss(irradiance_grad_x * weights, color_grad_x * weights)

    L2_loss = nn.functional.mse_loss(irradiance * mask, color * mask)

    shading_loss = L2_loss + grad_weight * grad_loss

    return shading_loss, irradiance, albedo


def fidelity_loss(depth_pred, depth_gt, mask, L1=0.0, L2=1.0):
    ret = 0
    if L1 > 0.0:
        ret += L1 * nn.functional.l1_loss(depth_pred * mask, depth_gt * mask)
    if L2 > 0.0:
        ret += L2 * nn.functional.mse_loss(depth_pred * mask, depth_gt * mask)

    return ret


def smoothness_loss(depth_pred, mask):
    # diff_1 = torch.sum(torch.abs(depth_pred[:, :, 1:-1, 1:-1] - depth_pred[:, :, 0:-2, 1:-1]) * mask[:, :, 1:-1, 1:-1])
    # diff_2 = torch.sum(torch.abs(depth_pred[:, :, 1:-1, 1:-1] - depth_pred[:, :, 1:-1, 0:-2]) * mask[:, :, 1:-1, 1:-1])
    ret = kornia.losses.total_variation(depth_pred * mask, reduction='mean').mean()
    return ret


def RefineNetLoss(depth_gt, depth_pred, normals_gt, normals_pred, mask, rgb, alb):
    _, c, h, w = depth_gt.shape
    lambda_sh = 5
    lambda_fid = 2
    lambda_smooth = 0.0001

    shadingLoss, _, _ = shading_loss(normals_pred, normals_gt, rgb, mask, albedo=alb, grad_metric=True,
                                     grad_weight=100.0, avg_pool=False)
    fidelityLoss = fidelity_loss(depth_pred, depth_gt, mask * 65535 / 2.0, L1=1.0, L2=1.0)
    smoothLoss = smoothness_loss(depth_pred, mask)
    # print(shadingLoss)
    # print(fidelityLoss)
    # print(smoothLoss)
    # exit()
    ret = lambda_sh * shadingLoss + lambda_fid * fidelityLoss + lambda_smooth * smoothLoss
    return ret

# def get_lighting_coefficiants(normals, image, abd, mask, rm_graz=False, eps=1e-4):
#     # print(torch.unique(normals[0, 2:3, :, :]))
#     print(normals.shape)
#     print(image.shape)
#     print(abd.shape)
#     exit()
#     if rm_graz:
#         mask_grazed = torch.abs(normals[:, 2:3, :, :]) > 0.3
#         mask = mask * mask_grazed
#     # fig, ax = plt.subplots(1, 4)
#     # ax[0].imshow(mask.squeeze()[0].cpu().detach().numpy())
#     # ax[1].imshow(mask_grazed.squeeze()[0].cpu().detach().numpy())
#     # ax[2].imshow(image.squeeze()[0].cpu().detach().numpy())
#     # ax[3].imshow(normals[0, 2, :, :].cpu().detach().numpy())
#     # plt.show()
#     # exit()
#     b, c, h, w = image.shape
#     image = torch.permute(torch.reshape(image, [-1, c, h * w]), [0, 2, 1])  # B x H*W x 1
#     A = get_spherical_harmonics(normals, mask)  # B x 9 x H*W
#
#     if abd is not None:
#         abd = torch.reshape(abd, [-1, 1, h * w])
#         A = A * abd
#
#     A_t = torch.permute(A, dims=[0, 2, 1])  # B x H*W x 9
#     AA_t = torch.matmul(A, A_t) + eps * torch.eye(9).cuda()  # B x 9 x 9
#
#     lighting_coeffs = torch.matmul(torch.matmul(torch.linalg.inv(AA_t), A), image).squeeze(2)
#     LH = torch.reshape(torch.matmul(lighting_coeffs.unsqueeze(1), A), [-1, 1, h, w])
#
#     return lighting_coeffs, LH, mask


# def get_lighting_coefficiants(normals, image, abd, mask, rm_graz=False, eps=1e-4):
#     b, c, h, w = image.shape
#     if rm_graz:
#         mask_grazed = torch.abs(normals[:, 2:3, :, :]) > 0.3
#         mask = mask * mask_grazed
#
#     A = get_spherical_harmonics(normals, mask)  # B x 9 x H*W
#     image = torch.permute(torch.reshape(image, [-1, c, h * w]), [0, 2, 1])
#     if abd is not None:
#         abd = torch.reshape(abd, [-1, c, h * w])
#         print(abd.shape)
#         abd = torch.tile(abd.unsqueeze(2), (1, 1, 9, 1))  # B x C x 9 x H*W
#         print(abd.shape)
#         A = A.unsqueeze(1) * abd  # B x C x 9 x H*W
#         print(A.shape)
#
#     A_t = torch.permute(A, dims=[0, 1, 3, 2])  # B x C x H*W x 9
#     AA_t = torch.matmul(A, A_t) + eps * torch.eye(9).cuda()  # B x C x 9 x 9
#
#     print(AA_t.shape)
#     print(image.shape)
#     print(torch.matmul(torch.linalg.inv(AA_t), A).shape)
#     exit()
#     lighting_coeffs = torch.matmul(torch.matmul(torch.linalg.inv(AA_t), A), image).squeeze(2)
#     print(lighting_coeffs.shape)
#     exit()
#     LH = torch.reshape(torch.matmul(lighting_coeffs.unsqueeze(1), A), [-1, c, h, w])
#
#     return lighting_coeffs, LH, mask
