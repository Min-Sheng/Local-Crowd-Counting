import os
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

from misc.utils import *
from test_config import cfg

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


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


image_dir = './demo_image'
data_path = './slide_examples.json'

def main():
    with open(data_path) as f:
        file_list = json.load(f)
    test(file_list, model_path)


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
        ori_img = Image.open(imgname)
        if ori_img.mode == 'L':
            ori_img = ori_img.convert('RGB')
        img = img_transform(ori_img)

        start_time = time.time()
        with torch.no_grad():
            img = Variable(img[None, :, :, :]).cuda()
            pred_map = net.test_forward(img)
        inference_time = time.time() - start_time
        print(f'Inference time: {inference_time:.3f}')
        inference_times.append(inference_time)
            
        ''' MAE/MSE'''
        pred_value = np.sum(pred_map.cpu().data.numpy()[0, 0, :, :]) / cfg.LOG_PARA 
        print("count is:", pred_value)
        
        
        ''' pred counting map '''
        den_frame = plt.gca()
        image = pred_map.cpu().data.numpy()[0, 0, :, :]
        image = cv2.resize(image, (image.shape[1]*8, image.shape[0]*8))
        plt.imshow(np.array(ori_img))
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