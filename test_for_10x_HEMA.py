import os
import glob
import cv2
import math
import json
import pandas as pd
from PIL import Image
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as standard_transforms

import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

from misc.utils import *
from test_config import cfg

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

plot = False
''' prepare model config '''
model_net = cfg.NET
model_path = cfg.MODEL_PATH

cfg_GPU_ID = cfg.GPU_ID
torch.cuda.set_device(cfg_GPU_ID[0])
torch.backends.cudnn.benchmark = True


''' prepare data config '''
data_mode = cfg.DATASET
from datasets.HEMA.setting import cfg_data
    
mean_std = cfg_data.MEAN_STD
img_transform = standard_transforms.Compose([
    standard_transforms.ToTensor(),
    standard_transforms.Normalize(*mean_std)
])


image_dir = './hema_10x_res'
if not os.path.exists(image_dir):
    os.mkdir(image_dir)
file_list = sorted(glob.glob('/mnt/dataset/NTUH_HEMA/Original_Image/*/*_BM_10x_*.tiff'))

def main():
    test(file_list, model_path)

def get_local_maximum(pred):
    neighborhood_size = 3
    threshold = 10
    
    data_max = filters.maximum_filter(pred, neighborhood_size)
    maxima = (pred == data_max)
    data_min = filters.minimum_filter(pred, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    x, y = [], []
    for dy,dx in slices:
        x_center = (dx.start + dx.stop - 1)/2
        x.append(x_center)
        y_center = (dy.start + dy.stop - 1)/2    
        y.append(y_center)
    return x, y

def test(file_list, model_path):
    ''' model '''
    from models.CC_DM import CrowdCounter
    net = CrowdCounter(cfg_GPU_ID, model_net, pretrained=False)
    
    ''' single-gpu / multi-gpu trained model '''
    if len(cfg_GPU_ID) == 1:
        net.load_state_dict(torch.load(model_path, map_location='cuda:0'))
    else:
        load_gpus_to_gpu(net, model_path)
    net.cuda()
    net.eval()

    index = 0
    inference_times = []
    for filename in file_list:
        index += 1
        print(index, filename)

        # read img
        imgname = filename
        
        # model testing
        # ori_img = Image.open(imgname)
        # if ori_img.mode == 'L':
        #     ori_img = ori_img.convert('RGB')
        np_ori_img = cv2.imread(imgname)[...,::-1]
        np_ori_img = cv2.resize(np_ori_img, (int(np_ori_img.shape[1]*0.63), int(np_ori_img.shape[0]*0.63)))
        ori_img = Image.fromarray(np_ori_img)

        img = img_transform(ori_img)

        start_time = time.time()
        with torch.no_grad():
            img = Variable(img[None, :, :, :]).cuda()
            pred_map = net.test_forward(img)
        inference_time = time.time() - start_time
        print(f'Inference time: {inference_time:.3f}')
        inference_times.append(inference_time)
            
        ''' MAE/MSE'''
        pred_map = pred_map.cpu().data.numpy()[0, 0, :, :]
        pred_value = np.sum(pred_map) / cfg.LOG_PARA 
        print("count is:", pred_value)

        x, y = get_local_maximum(pred_map)
        x_resized = [i*8 for i in x]
        y_resized = [i*8 for i in y]
        centers = np.array([x_resized, y_resized]).T

        post_processor = PostProcessor()
        post_processor.set_global_anchor(np_ori_img, image_color_space='rgb')
        post_processor.run(np_ori_img)
        
        filtered_center = []
        for c in centers:
            if post_processor.gray_mask[int(c[1]), int(c[0])] == False:
                filtered_center.append(c)
        filtered_center = np.array(filtered_center)

        np.savetxt(os.path.join(image_dir, os.path.basename(imgname).split('.')[0] + '.csv'), centers, fmt='%i', delimiter=",")

        np.savetxt(os.path.join(image_dir, os.path.basename(imgname).split('.')[0] + '_filtered.csv'), filtered_center, fmt='%i', delimiter=",")

        if plot:
            ''' pred counting map '''
            den_frame = plt.gca()
            plt.imshow(np.array(ori_img))
            image = cv2.resize(image, (image.shape[1]*8, image.shape[0]*8))
            plt.imshow(image, 'jet', alpha=0.45)
            den_frame.axes.get_yaxis().set_visible(False)
            den_frame.axes.get_xaxis().set_visible(False)
            den_frame.spines['top'].set_visible(False)
            den_frame.spines['bottom'].set_visible(False)
            den_frame.spines['left'].set_visible(False)
            den_frame.spines['right'].set_visible(False)
        
            save_dir = os.path.join(image_dir, 'result')
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            plt.savefig(os.path.join(save_dir, os.path.basename(imgname).split('.')[0] + '_predmap_' + str(int(pred_value + 0.5)) + '.jpg'),
                        bbox_inches='tight', pad_inches=0, dpi=1024)
    print(f'Averge inference time: {np.mean(inference_times):.3f}')
            
if __name__ == '__main__':
    
    main()