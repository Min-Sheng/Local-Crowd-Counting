import os
from easydict import EasyDict as edict
import time
import torch

# init
__C = edict()
cfg = __C

#------------------------------TRAIN------------------------
# if train batch=1, use only one GPU!!
__C.GPU_ID = [0] 		    # sigle gpu: [0], [1] ...; multi gpus: [0,1]

__C.NET = 'CSRNet_DM' 	    
                            # DM model: CSRNet_DM, VGG16_DM, Res50_DM
                            # LCM model: CSRNet_LCM, VGG16_LCM, VGG16_LCM_REG

__C.DATASET = 'HEMA' 	    # SHHA, SHHB_C, UCF50, QNRF,

__C.HEMAPATCHMAX = 50. #30.
__C.SHHBPATCHMAX = 30.
__C.SHHAPATCHMAX = 100.
__C.QNRFPATCHMAX = 100.
__C.CC50PATCHMAX = 100.

__C.SEED = 3035 		    # random seed, for reproduction

__C.LOG_PARA = 100.

if __C.DATASET == 'UCF50':	# only for UCF50
    from datasets.UCF50.setting import cfg_data
    __C.VAL_INDEX = cfg_data.VAL_INDEX


__C.PRE_GCC = False 		# use the pretrained model on GCC dataset
__C.PRE_GCC_MODEL = 'path to model' 	# path to model

__C.RESUME = False 			# contine training
__C.RESUME_PATH = ''


# learning rate settings
__C.LR = 1e-5 				    # learning rate
__C.LR_DECAY = 0.95   		    # decay rate 0.95
__C.LR_DECAY_START = -1 	    # when training epoch is more than it, the learning rate will be begin to decay
__C.NUM_EPOCH_LR_DECAY = 200 	# decay frequency
__C.MAX_EPOCH = 500

# multi-task learning weights, no use for single model, such as MCNN, VGG, VGG_DECODER, Res50, CSRNet, and so on

__C.LAMBDA_1 = 1e-4			# SANet:0.001 CMTL 0.0001


# print 
__C.PRINT_FREQ = 50

now = time.strftime("%m-%d_%H-%M", time.localtime())

__C.EXP_NAME = now \
			 + '_' + __C.DATASET \
             + '_' + __C.NET \
             + '_' + str(__C.LR)

if __C.DATASET == 'UCF50':
    __C.EXP_NAME += '_' + str(__C.VAL_INDEX)

__C.EXP_PATH = './exp' # the path of logs, checkpoints, and current codes


#------------------------------VAL------------------------
__C.VAL_DENSE_START = 0
__C.VAL_FREQ = 10 			# Before __C.VAL_DENSE_START epoches, the freq is set as __C.VAL_FREQ

#------------------------------VIS------------------------
__C.VISIBLE_NUM_IMGS = 1 	# must be 1 for training images with the different sizes



#================================================================================
#================================================================================  
