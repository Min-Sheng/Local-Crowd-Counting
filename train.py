import os
import numpy as np
import torch

from train_config import cfg

''' prepare enviroment '''
seed = cfg.SEED
if seed is not None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

gpus = cfg.GPU_ID
torch.cuda.set_device(gpus[0])
torch.backends.cudnn.benchmark = True


''' prepare data loader '''
data_mode = cfg.DATASET
if data_mode is 'SHHA':
    from datasets.SHHA.loading_data import loading_data 
    from datasets.SHHA.setting import cfg_data 
elif data_mode is 'SHHB':
    from datasets.SHHB.loading_data import loading_data 
    from datasets.SHHB.setting import cfg_data 
elif data_mode is 'QNRF':
    from datasets.QNRF.loading_data import loading_data 
    from datasets.QNRF.setting import cfg_data 
elif data_mode is 'UCF50':
    from datasets.UCF50.loading_data import loading_data
    from datasets.UCF50.setting import cfg_data 
elif data_mode is 'HEMA':
    from datasets.HEMA.loading_data import loading_data
    from datasets.HEMA.setting import cfg_data 
elif data_mode is 'HEMA_10x':
    from datasets.HEMA_10x.loading_data import loading_data
    from datasets.HEMA_10x.setting import cfg_data 



''' Prepare Trainer '''
net = cfg.NET
if net in ['CSRNet_DM', 'VGG16_DM', 'Res50_DM']:
    from models.trainer_for_DM import Trainer
    
elif net in ['CSRNet_LCM', 'VGG16_LCM', 'VGG16_LCM_REG']:
    from models.trainer_for_LCM import Trainer


''' Start Training '''
pwd = os.path.split(os.path.realpath(__file__))[0]
cc_trainer = Trainer(loading_data, cfg_data, pwd)
cc_trainer.forward()
