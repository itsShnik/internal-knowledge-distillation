#----------------------------------------
#--------- Torch Related Imports --------
#----------------------------------------
import torch

#----------------------------------------
#--------- OS and Python Lib Imports ----
#----------------------------------------
import os
import wandb
import argparse

#----------------------------------------
#--------- Module Related ---------------
#----------------------------------------
from modules.networks_for_cifar_resnet import *
from modules.networks_for_original_resnet import *
from modules.networks_for_modified_resnet import *

#----------------------------------------
#--------- Common and utils -------------
#----------------------------------------
from functions.config import config, update_config
from common.utils import smart_model_load
from functions.val import do_validation
from dataloaders.build import make_dataloader, build_dataset

def parse_args():
    parser = argparse.ArgumentParser('Inference Script')
    parser.add_argument('--cfg', type=str, help='path to config file')

    args = parser.parse_args()

    if args.cfg is not None:
        update_config(args.cfg)

    return args, config

def main():
    """
    Parse the arugemnts and get the config file
    """
    args, config = parse_args()

    """
    Log on wandb for track of experiments
    """
    wandb.init(project="adaptive-finetuning-resnet", name=f'Inference_{config.VERSION}', config=config)

    """
    Set config GPUs and torch cuda device
    """
    config.GPUS = str(0)
    torch.cuda.set_device(0)

    """
    Create the model, put it to GPU and then create dataloader
    """
    model = eval(config.MODULE)(config=config.NETWORK)
    model = model.cuda()

    val_loader = make_dataloader(config, mode='val', distributed=False)

    """
    Load the model with pretrained weights
    """
    assert config.NETWORK.PRETRAINED_MODEL != '', "For inference, there must be pre-trained weights"

    pretrain_state_dict = torch.load(config.NETWORK.PRETRAINED_MODEL, map_location = lambda storage, loc: storage)['net_state_dict']
    smart_model_load(model, pretrain_state_dict, loading_method=config.NETWORK.PRETRAINED_LOADING_METHOD)

    """
    Pass the model and val loader for validation
    """
    print("Inference started!!")
    val_accuracy = do_validation(config, model, val_loader)
    print(f"Inference complete!!\nAccuracy:{val_accuracy}")

    wandb.log({'Accuracy': val_accuracy})

if __name__=='__main__':
    main()
