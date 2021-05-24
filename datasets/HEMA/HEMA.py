import numpy as np
import json
import random
import torch
from torch.utils import data
import cv2
from PIL import Image

class HEMA(data.Dataset):
    def __init__(self, data_path, main_transform=None, img_transform=None, gt_transform=None,
                 data_augment=1, k_size=9, sigma=2, exclude_invalid=False):
        # Fetch training and validation subsets
        with open(data_path) as f:
            self.data_files = json.load(f)
        self.num_samples = len(self.data_files)
        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.data_augment = data_augment
        self.k_size = k_size
        self.sigma = sigma
        self.exclude_invalid = exclude_invalid

    def __getitem__(self, index):
        # for data_augment
        index = int(index/self.data_augment)

        fname = self.data_files[index]
        img, den = self.read_image_and_gt(fname)
        if self.main_transform is not None:
            img, den = self.main_transform(img, den)
        if self.img_transform is not None:
            img = self.img_transform(img)
        if self.gt_transform is not None:
            den = self.gt_transform(den)

        return img, den

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self, fname):
        # Load image
        img = Image.open(fname)
        if img.mode == 'L':
            img = img.convert('RGB')

        # Load density map
        # if self.k_size == 15 and self.sigma == 4:
        #     if self.exclude_invalid:
        #         target_path = fname.replace('.tiff', f'_density_k{self.k_size}_sigma{self.sigma}_exclude_invalid.npy')
        #     else:
        #         target_path = fname.replace('.tiff', '_density.npy')
        # else:
        if self.exclude_invalid:
            target_path = fname.replace('.tiff', f'_density_k{self.k_size}_sigma{self.sigma}_exclude_invalid.npy')
        else:
            target_path = fname.replace('.tiff', f'_density_k{self.k_size}_sigma{self.sigma}.npy')

        den = np.load(target_path)
        den = den.astype(np.float32, copy=False)
        den = Image.fromarray(den)

        return img, den

    def get_num_samples(self):
        return self.num_samples
