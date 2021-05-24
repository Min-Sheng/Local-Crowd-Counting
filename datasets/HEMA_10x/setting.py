from easydict import EasyDict as edict

# init
__C_HEMA_10x = edict()

cfg_data = __C_HEMA_10x
__C_HEMA_10x.TRAIN_SIZE = (456, 684)

__C_HEMA_10x.DATA_PATH = './ProcessedData/hema_10x'

# __C_HEMA_10x.MEAN_STD = ([0.410824894905, 0.370634973049, 0.359682112932],
#                      [0.278580576181, 0.26925137639, 0.27156367898])

__C_HEMA_10x.MEAN_STD = ([0.7389957904815674, 0.7036832571029663, 0.7821053862571716],
                     [0.2377883344888687, 0.2641891837120056, 0.15579012036323547])

# __C_HEMA_10x.MEAN_STD = ([0.485, 0.456, 0.406],
#                      [0.229, 0.224, 0.225])

__C_HEMA_10x.LABEL_FACTOR = 1
__C_HEMA_10x.LOG_PARA = 100.

__C_HEMA_10x.RESUME_MODEL = ''      # model path

__C_HEMA_10x.TRAIN_BATCH_SIZE = 16   # must be 1
__C_HEMA_10x.VAL_BATCH_SIZE = 1     # must be 1
