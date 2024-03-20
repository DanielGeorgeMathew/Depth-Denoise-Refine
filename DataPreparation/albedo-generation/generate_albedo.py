from solver import IntrinsicSolver
from input import IntrinsicInput
from params import IntrinsicParameters
import image_util

import numpy as np
import os
import cv2
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import json
import random

train_ids = [1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13]

for id in train_ids:
    root_path = '~/Downloads/DDRNet-TrainData/DDRNet-TrainData/scan-{}.xcappdata/AppData/tmp'.format(id)
    rgb_files = sorted(glob(os.path.join(root_path, 'rgb/*.png')), key=lambda x: int(os.path.basename(x).split('.')[0]))
    target_path = os.path.join(root_path, 'albedo')
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    for i in tqdm(rgb_files[::5]):
        input = IntrinsicInput.from_file(i)
        solver = IntrinsicSolver(input, {'n_iters': 10})
        r, s, decomposition = solver.solve()
        # plt.imshow(r)
        # plt.show()
        r = (r * 255).astype('uint8')
        # plt.imshow(r)
        # plt.show()
        r = cv2.cvtColor(r, cv2.COLOR_BGR2RGB)
        # plt.imshow(r)
        # plt.show()
        # r[:, :, 0], r[:, :, 2] = r[:, :, 2], r[:, :, 0]
        # np.save(os.path.join(target_path, os.path.basename(i).split('.')[0]+'.npy'), r)
        # plt.imshow(r)
        # plt.show()
        cv2.imwrite(os.path.join(target_path, os.path.basename(i)), r)
