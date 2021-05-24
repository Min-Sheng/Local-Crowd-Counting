import os
import numpy as np
import json
import random
import torch
from torch.utils import data
import cv2
from PIL import Image

class HEMA_10x(data.Dataset):
    def __init__(self, data_path, main_transform=None, img_transform=None, gt_transform=None,
                 data_augment=1):
        # Fetch training and validation subsets
        self.img_path = data_path + '/img' 
        self.gt_path = data_path + '/den'
        self.data_files = [filename for filename in os.listdir(self.img_path) \
                           if os.path.isfile(os.path.join(self.img_path,filename))]        
        self.num_samples = len(self.data_files)
        self.main_transform = main_transform
        self.img_transform = img_transform
        self.gt_transform = gt_transform
        self.data_augment = data_augment

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
        img = Image.open(os.path.join(self.img_path, fname))
        if img.mode == 'L':
            img = img.convert('RGB')

        # Load density map
        target_path = os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.npy')
        den = np.load(target_path)
        den = den.astype(np.float32, copy=False)
        den = Image.fromarray(den)

        return img, den

    def get_num_samples(self):
        return self.num_samples
