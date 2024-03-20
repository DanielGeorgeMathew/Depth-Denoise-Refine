import torch
from torch import nn
from ModelPreparation.loss.loss_functions_RefineNet import RefineNetLoss
from ModelPreparation.loss.loss_functions_DenoiseNet import DenoiseNetLoss
import matplotlib.pyplot as plt

def getDepthReal(depth_in):
    low_thresh = 0
    up_thresh = 65535
    thresh_range = (up_thresh-low_thresh)/2.0
    ret = ((depth_in + 1)*thresh_range)/30000.0
    return ret


def depthToNormals(depth, A):
    b, h, w = depth.shape
    fx = A[0][0]
    fy = A[1][1]
    cx = A[0][2]
    cy = A[1][2]
    inv_fx = 1.0 / fx
    inv_fy = 1.0 / fy

    v_coord, u_coord = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    U_coord = torch.tile(u_coord.unsqueeze(0), [b, 1, 1]).cuda().float()
    V_coord = torch.tile(v_coord.unsqueeze(0), [b, 1, 1]).cuda().float()

    X_coord = -(U_coord - cx) * inv_fx * depth
    Y_coord = -(V_coord - cy) * inv_fy * depth
    points3d = torch.stack([X_coord, Y_coord, depth], dim=3)

    p_ctr = points3d[:, 1:-1, 1:-1, :]
    vw = p_ctr - points3d[:, 1:-1, 2:, :]
    vs = points3d[:, 2:, 1:-1, :] - p_ctr
    ve = p_ctr - points3d[:, 1:-1, :-2, :]
    vn = points3d[:, :-2, 1:-1, :] - p_ctr

    normal_1 = torch.cross(vs, vw, dim=3)
    normal_2 = torch.cross(vn, ve, dim=3)
    normal_1 = torch.nn.functional.normalize(normal_1, p=2, dim=3)
    normal_2 = torch.nn.functional.normalize(normal_2, p=2, dim=3)
    normal = normal_1 + normal_2

    normal = torch.nn.functional.normalize(normal, p=2, dim=3)
    paddings = (0, 0, 1, 1, 1, 1)
    normal = torch.nn.functional.pad(normal, paddings, mode='constant')  # BxHxWx3
    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(depth[0].detach().cpu().numpy())
    # ax[1].imshow(normal[0, :, :, 2].detach().cpu().numpy())
    # plt.show()
    return torch.permute(normal, [0, 3, 1, 2])


def DDRNet_Loss(depth_gt, depth_denoised, depth_refined, mask, rgb, A, alb, orig):
    normals_gt = depthToNormals(getDepthReal(depth_gt.squeeze()), A)
    normals_denoised = depthToNormals(getDepthReal(torch.clamp(depth_denoised, -1, 1).squeeze()), A)
    normals_refined = depthToNormals(getDepthReal(depth_refined.squeeze()), A)

    lambda_refine = 1

    denoiseLoss = DenoiseNetLoss(depth_gt, depth_denoised, normals_gt, normals_denoised, mask, A, orig)
    refineLoss = RefineNetLoss(depth_gt, depth_refined, normals_gt, normals_refined, mask, rgb, alb)

    ret = denoiseLoss + lambda_refine*refineLoss




    return ret
