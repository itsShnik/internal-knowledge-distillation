#----------------------------------------
#--------- Python Lib Related Imports ---
#----------------------------------------
from easydict import EasyDict as edict
import yaml

# Create an edict for config
_C = edict()
config = _C

#----------------------------------------
#--------- Common Options ---------------
#----------------------------------------
_C.RNG_SEED = -1
_C.MODULE = ''
_C.GPUS = ''
_C.NUM_WORKERS_PER_GPU = 4
_C.VERSION = 'adaptive-finetune'
_C.NUM_CLASSES = 1000
_C.TRAINING_STRATEGY = 'standard'
_C.POLICY_MODULE = 'resnet8'

#----------------------------------------
#--------- Dataset related options ------
#----------------------------------------
_C.DATASET = edict()
_C.DATASET.DATASET_NAME  = ''
_C.DATASET.ROOT_PATH  = ''
_C.DATASET.TRAIN_SPLIT  = ''
_C.DATASET.VAL_SPLIT  = ''
_C.DATASET.TOY = False

#----------------------------------------
#--------- Network Related Options ------
#----------------------------------------
_C.NETWORK = edict()
_C.NETWORK.PRETRAINED_MODEL = ''

#----------------------------------------
#--------- Policy Related Options -------
#----------------------------------------
_C.POLICY = edict()
_C.POLICY.OPTIMIZER = 'SGD'
_C.POLICY.LR = 0.01
_C.POLICY.LR_STEPS = [30, 60, 90]
_C.POLICY.LR_DECAY = 0.1
_C.POLICY.MOMENTUM = 0.9
_C.POLICY.WEIGHT_DECAY = 0.001

#----------------------------------------
#--------- Training related options -----
#----------------------------------------
_C.TRAIN = edict()
_C.TRAIN.BATCH_IMAGES = 128
_C.TRAIN.LR = 5e-4
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WEIGHT_DECAY = 1e-4
_C.TRAIN.OPTIMIZER = 'SGD'
_C.TRAIN.LR_STEPS = [30, 60, 90]
_C.TRAIN.LR_DECAY = 0.1
_C.TRAIN.ASPECT_GROUPING = False
_C.TRAIN.SHUFFLE = False
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.END_EPOCH = 110

#----------------------------------------
#--------- Validation related options ---
#----------------------------------------
_C.VAL = edict()
_C.VAL.BATCH_IMAGES = 128
_C.VAL.SHUFFLE = False

def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    for vk, vv in v.items():
                        if vk in config[k]:
                            config[k][vk] = vv
                        else:
                            raise ValueError("key {}.{} not in config.py".format(k, vk))
                else:
                    config[k] = v
            else:
                raise ValueError("key {} not in config.py".format(k))
