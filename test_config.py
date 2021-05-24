import os
from easydict import EasyDict as edict
import time
import torch

# init
__C = edict()
cfg = __C

# ------------------------------TEST------------------------
# if train batch=1, use only one GPU!!
__C.GPU_ID = [0] 		    # sigle gpu: [0], [1] ...; multi gpus: [0,1]

__C.NET = 'CSRNet_DM' 	# net selection as train_config.py

__C.DATASET = 'HEMA' 	    # SHHA, SHHB, QNRF, UCF50

__C.HEMAPATCHMAX = 30.
__C.SHHBPATCHMAX = 30.
__C.SHHAPATCHMAX = 100.
__C.QNRFPATCHMAX = 100.
__C.CC50PATCHMAX = 100.
                            # testing model path
#__C.MODEL_PATH = './exp/05-04_04-49_HEMA_CSRNet_DM_1e-05/all_ep_214_mae_2.00_mse_4.28.pth'

#__C.MODEL_PATH = './exp/05-06_03-43_HEMA_CSRNet_DM_1e-05/all_ep_362_mae_2.19_mse_4.57.pth'

#__C.MODEL_PATH = './exp/05-12_07-45_HEMA_10x_CSRNet_DM_1e-05/all_ep_217_mae_30.30_mse_50.62.pth'

__C.MODEL_PATH = './exp/05-14_08-02_HEMA_CSRNet_DM_1e-05/all_ep_447_mae_2.21_mse_4.61.pth'

__C.LOG_PARA = 100.

if __C.DATASET == 'UCF50':  # only for UCF50
    from datasets.UCF50.setting import cfg_data
    __C.VAL_INDEX = cfg_data.VAL_INDEX
