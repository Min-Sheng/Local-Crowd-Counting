from easydict import EasyDict as edict

# init
__C_HEMA = edict()

cfg_data = __C_HEMA
__C_HEMA.TRAIN_SIZE = (230, 345)
__C_HEMA.K_SIZE = 27 #15 #11
__C_HEMA.SIGMA = 5 # 2
__C_HEMA.EXCLUDE_INVALID = False

__C_HEMA.TRAIN_DATA_PATH = './ProcessedData/hema/train_63x.json'
__C_HEMA.VAL_DATA_PATH = './ProcessedData/hema/val_63x.json'
__C_HEMA.TEST_DATA_PATH = './ProcessedData/hema/test_63x.json'

# __C_HEMA.MEAN_STD = ([0.410824894905, 0.370634973049, 0.359682112932],
#                      [0.278580576181, 0.26925137639, 0.27156367898])

__C_HEMA.MEAN_STD = ([0.7389957904815674, 0.7036832571029663, 0.7821053862571716],
                     [0.2377883344888687, 0.2641891837120056, 0.15579012036323547])

# __C_HEMA.MEAN_STD = ([0.485, 0.456, 0.406],
#                      [0.229, 0.224, 0.225])

__C_HEMA.LABEL_FACTOR = 1
__C_HEMA.LOG_PARA = 100.

__C_HEMA.RESUME_MODEL = ''      # model path

__C_HEMA.TRAIN_BATCH_SIZE = 16   # must be 1
__C_HEMA.VAL_BATCH_SIZE = 1     # must be 1
