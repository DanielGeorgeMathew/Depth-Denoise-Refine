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
    RandomCrop

import open3d as o3d


import coremltools as ct


device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
weights_path = '~/Daniel/DepthRefinement/checkpoints/DDRNet-v3'

d_model = convResnet()
r_model = hyperColumn()
d_model.to(device)
r_model.to(device)

d_model.load_state_dict(torch.load(os.path.join(weights_path, 'denoise_model.pt'), map_location=device))
r_model.load_state_dict(torch.load(os.path.join(weights_path, 'refine_model.pt'), map_location=device))

d_model.eval()
r_model.eval()

dummy_input_1 = torch.rand(1, 1, 640, 480).to(device)
dummy_input_2 = torch.rand(1, 1, 640, 480).to(device)
dummy_input_3 = torch.rand(1, 3, 640, 480).to(device)

traced_d_model = torch.jit.trace(d_model, dummy_input_1)
traced_r_model = torch.jit.trace(r_model, (dummy_input_2, dummy_input_3))

denoised_out = traced_d_model(dummy_input_1)
refined_out = traced_r_model(dummy_input_2, dummy_input_3)

high_thresh = 65535
low_thresh = 0
thresh_range = (high_thresh - low_thresh)/2.0


d_mlmodel = ct.convert(traced_d_model,
                       inputs=[ct.ImageType(name='input-depth', shape=(1, 1, 640, 480), scale=1/thresh_range, bias=-1.0, color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
                       outputs=[ct.ImageType(name='denoised-depth', color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
                       compute_units=ct.ComputeUnit.ALL,
                       minimum_deployment_target=ct.target.iOS16)




d_mlmodel.save('~/Desktop/denoiser_v3.mlpackage')


r_mlmodel = ct.convert(traced_r_model,
                       inputs=[ct.ImageType(name='denoised-depth',
                                            shape=(1, 1, 640, 480),
                                            color_layout=ct.colorlayout.GRAYSCALE_FLOAT16),
                               ct.ImageType(name='input-color',
                                            shape=(1, 3, 640, 480),
                                            scale=1 / 127.0,
                                            bias=[-1.0, -1.0, -1.0],
                                            color_layout=ct.colorlayout.RGB)
                               ],
                       outputs=[ct.ImageType(name='refined-depth', color_layout=ct.colorlayout.GRAYSCALE_FLOAT16)],
                       compute_units=ct.ComputeUnit.ALL,
                       minimum_deployment_target=ct.target.iOS16)

r_mlmodel.save('~/Desktop/refiner_v3.mlpackage')

