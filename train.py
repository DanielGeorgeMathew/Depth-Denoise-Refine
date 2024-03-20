import argparse
import json
import os
import random

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torchvision.transforms import Compose

from ModelPreparation.models_DDRNet.model_denoise import convResnet
from ModelPreparation.models_DDRNet.model_refine import hyperColumn
from ModelPreparation.loss.loss_main import DDRNet_Loss
from ModelPreparation.data.datasets import sureScanDataset
from ModelPreparation.models.util import EarlyStopping
from ModelPreparation.data.transforms import Scale, RandomHorizontalFlip, CenterCrop, ToTensor, ColorJitter, Normalize, \
    RandomCrop, RandomBrightness



def test_sample(d_model, r_model, val_dataset, i):
    with torch.no_grad():
        up_thresh = 65535
        low_thresh = 0
        thresh_range = (up_thresh - low_thresh) / 2.0
        d_model.eval()
        r_model.eval()
        fig, ax = plt.subplots(1, 5, figsize=(10, 10))

        random_index = int(np.random.random() * n_test)
        test_dict = val_dataset[random_index]

        test_img_unnormalized = torch.permute((test_dict['rgb']+1)*127.0,
                                              (1, 2, 0)).cpu().numpy().astype(int)
        denoised_output = d_model(test_dict['depth'].cuda().unsqueeze(0))
        refined_output = r_model(denoised_output, test_dict['rgb'].cuda().unsqueeze(0))
        denoised_output = denoised_output.squeeze().cpu().numpy()
        refined_output = refined_output.squeeze().cpu().numpy()
        denoised_output = (denoised_output + 1) * thresh_range
        refined_output = (refined_output + 1) * thresh_range
        mask = test_dict['mask'].squeeze().numpy()
        gt_depth = test_dict['gt_depth'].squeeze().numpy()
        depth = test_dict['depth'].squeeze().numpy()
        gt_depth = (gt_depth + 1) * thresh_range
        depth = (depth + 1) * thresh_range
        ax[0].set_title("RGB", fontsize=fontsize)
        ax[0].imshow(test_img_unnormalized)

        ax[1].set_title("ORIGINAL DEPTH", fontsize=fontsize)
        ax[1].imshow(depth)

        ax[2].set_title("GT DEPTH", fontsize=fontsize)
        ax[2].imshow(gt_depth)

        ax[3].set_title("DENOISED DEPTH", fontsize=fontsize)
        ax[3].imshow(denoised_output*mask)

        ax[4].set_title("REFINED DEPTH", fontsize=fontsize)
        ax[4].imshow(refined_output*mask)
        # plt.show()
        writer.add_figure("RANDOM VALIDATION DURING TRAIN", fig, i)
        plt.close(fig)


REF_W, REF_H = 3024, 4032


def getExtrinsicIntrinsicFromMetadata(metadata_path):
    with open(metadata_path, 'r') as f:
        metadata_dict = json.load(f)
    extrinsic_list = np.asarray([i['matrix'] for i in metadata_dict['extrinsic']])
    intrinsic = np.asarray(metadata_dict['intrinsic']['matrix']).T
    intrinsic[0][0] = intrinsic[0][0] * 240 / REF_W
    intrinsic[1][1] = intrinsic[1][1] * 320 / REF_H
    intrinsic[0][2] = intrinsic[0][2] * 240 / REF_W
    intrinsic[1][2] = intrinsic[1][2] * 320 / REF_H

    return extrinsic_list, intrinsic


if __name__ == "__main__":

    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description='Depth Refinement Train')
    parser.add_argument('--data', type=str, default='')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--run_path', type=str, default='')
    parser.add_argument('--model_save_path', type=str, default='')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--metadata_path', type=str, default=None)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # _, intrinsic = getExtrinsicIntrinsicFromMetadata(args.metadata_path)
    intrinsic = np.load(args.metadata_path)

    # TRAIN HYPERPARAMETERS DEFINITIONS
    if not os.path.exists(args.model_save_path):
        os.makedirs(args.model_save_path)
    log_fp = open(os.path.join(args.model_save_path, 'logs.txt'), 'w+')
    fontsize = 5

    train_transforms = transforms.Compose([
        # RandomCrop((240, 320)),
        Scale((320, 240)),
        RandomBrightness(0.2),
        RandomHorizontalFlip(),
        ToTensor(),
    ])

    test_transforms = transforms.Compose([
        Scale((320, 240)),
        ToTensor(),
    ])

    all_transforms = {'train': train_transforms, 'val': test_transforms}
    train_dataset = sureScanDataset(os.path.join(args.data, 'train.txt'), all_transforms)
    val_dataset = sureScanDataset(os.path.join(args.data, 'val.txt'), all_transforms)
    writer = SummaryWriter(args.run_path)

    # INITIALIZING THE MODEL BELOW
    denoiseModel = convResnet()
    refineModel = hyperColumn()
    denoiseModel.to(device)
    refineModel.to(device)

    num_param_Denoise = sum(p.numel() for p in denoiseModel.parameters() if p.requires_grad)
    num_param_Refine = sum(p.numel() for p in refineModel.parameters() if p.requires_grad)
    print("Number of parameters in the Depth Denoise Model is {}".format(num_param_Denoise))
    print("Number of parameters in the Depth Refine Model is {}".format(num_param_Refine))

    best_accuracy = -1

    n_test = len(val_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               batch_sampler=None,
                                               num_workers=4, collate_fn=None, pin_memory=False, drop_last=True,
                                               timeout=0,
                                               worker_init_fn=None, multiprocessing_context=None, generator=None
                                               , prefetch_factor=2, persistent_workers=False)

    valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True,
                                               batch_sampler=None,
                                               num_workers=4, collate_fn=None, pin_memory=False, drop_last=True,
                                               timeout=0,
                                               worker_init_fn=None, multiprocessing_context=None, generator=None
                                               , prefetch_factor=2, persistent_workers=False)

    es = EarlyStopping(patience=args.patience, verbose=True, delta=0,
                       path=args.model_save_path)

    lr = 0.001
    optimizer = torch.optim.Adam(list(denoiseModel.parameters()) + list(refineModel.parameters()), lr=lr, weight_decay=0)

    criterion = DDRNet_Loss

    best_pred = 100000.0
    n_train = len(train_loader)
    for epoch in tqdm(range(args.num_epochs)):
        print("EPOCH {} is STARTING:".format(epoch))
        print("Training Started..............................")

        total_train_loss = 0
        total = 0
        denoiseModel.train()
        refineModel.train()
        with tqdm(train_loader) as tepoch:
            for i, data in enumerate(tepoch):
                rgb = data['rgb'].to(device)
                depth = data['depth'].to(device)
                gt_depth = data['gt_depth'].to(device)
                mask = data['mask'].to(device)
                albedo = data['albedo'].to(device)

                depth_denoised = denoiseModel(depth)
                depth_refined = refineModel(depth_denoised, rgb)

                train_loss = criterion(gt_depth, depth_denoised, depth_refined, mask, rgb, intrinsic, albedo, depth)

                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                total += 1
                total_train_loss += train_loss.item()
                tepoch.set_postfix(train_loss=train_loss.item())
                if (epoch * n_train + total) % 5 == 0:
                    test_sample(denoiseModel, refineModel, val_dataset, epoch * n_train + total)
            avg_train_loss = total_train_loss / total

        with torch.no_grad():
            print("Validation Started..............................")

            total_val_loss = 0
            total = 0
            denoiseModel.eval()
            refineModel.eval()
            with tqdm(valid_loader) as tepoch:
                for data in tepoch:
                    rgb = data['rgb'].to(device)
                    depth = data['depth'].to(device)
                    gt_depth = data['gt_depth'].to(device)
                    mask = data['mask'].to(device)
                    albedo = data['albedo'].to(device)
                    depth_denoised = denoiseModel(depth)
                    depth_refined = refineModel(depth_denoised, rgb)

                    val_loss = criterion(gt_depth, depth_denoised, depth_refined, mask, rgb, intrinsic, albedo, depth)

                    total += 1
                    total_val_loss += val_loss.item()
                    tepoch.set_postfix(val_loss=val_loss.item())

            avg_val_loss = total_val_loss / total
            if avg_val_loss < best_pred:
                best_pred = avg_val_loss
            es.evaluate(avg_val_loss, denoiseModel, refineModel)

        log_fp.write("Epoch {}/{}:  Train_Loss = {}, Val_Loss = {}\n".format(epoch, 800, avg_train_loss, avg_val_loss))
        print("Epoch {}/{}:  Train_Loss = {}, Val_Loss = {}".format(epoch, 800, avg_train_loss, avg_val_loss))

        writer.add_scalars('TRAINING PLOT', {'Training loss': avg_train_loss,
                                             'Validation loss': avg_val_loss
                                             }, epoch)
        with torch.no_grad():
            denoiseModel.eval()
            refineModel.eval()
            fig, ax = plt.subplots(1, 5, figsize=(10, 10))
            up_thresh = 65535
            low_thresh = 0
            thresh_range = (up_thresh - low_thresh) / 2.0
            random_index = int(np.random.random() * n_test)
            test_dict = val_dataset[random_index]

            test_img_unnormalized = torch.permute((test_dict['rgb'] + 1) * 127.0,
                                                  (1, 2, 0)).cpu().numpy().astype(int)
            denoised_output = denoiseModel(test_dict['depth'].cuda().unsqueeze(0))
            refined_output = refineModel(denoised_output, test_dict['rgb'].cuda().unsqueeze(0))
            denoised_output = denoised_output.squeeze().cpu().numpy()
            refined_output = refined_output.squeeze().cpu().numpy()
            denoised_output = (denoised_output + 1) * thresh_range
            refined_output = (refined_output + 1) * thresh_range
            mask = test_dict['mask'].squeeze().numpy()
            gt_depth = test_dict['gt_depth'].squeeze().numpy()
            depth = test_dict['depth'].squeeze().numpy()
            gt_depth = (gt_depth + 1) * thresh_range
            depth = (depth + 1) * thresh_range
            ax[0].set_title("RGB", fontsize=fontsize)
            ax[0].imshow(test_img_unnormalized)

            ax[1].set_title("ORIGINAL DEPTH", fontsize=fontsize)
            ax[1].imshow(depth)

            ax[2].set_title("GT DEPTH", fontsize=fontsize)
            ax[2].imshow(gt_depth)

            ax[3].set_title("DENOISED DEPTH", fontsize=fontsize)
            ax[3].imshow(denoised_output*mask)

            ax[4].set_title("REFINED DEPTH", fontsize=fontsize)
            ax[4].imshow(refined_output*mask)

            writer.add_figure("END OF EPOCH TEST", fig, epoch)
            plt.close(fig)

        if es.early_stop:
            print("Early Stopping at epoch {}".format(epoch))
            break
