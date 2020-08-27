#----------------------------------------
#--------- OS related imports -----------
#----------------------------------------
import os
import argparse
import subprocess

#----------------------------------------
#--------- Torch related imports --------
#----------------------------------------
import torch

#----------------------------------------
#--------- Config and training imports --
#----------------------------------------
from function.config import config, update_config
from function.train import train_net


def parse_args():
    parser = argparse.ArgumentParser('Train Cognition Network')
    parser.add_argument('--cfg', type=str, help='path to config file')
    parser.add_argument('--model-dir', type=str, help='root path to store checkpoint')
    parser.add_argument('--dist', help='whether to use distributed training', default=False, action='store_true')
    parser.add_argument('--cudnn-off', help='disable cudnn', default=False, action='store_true')

    args = parser.parse_args()

    if args.cfg is not None:
        update_config(args.cfg)

    return args, config


def main():
    args, config = parse_args()

    train_net(args, config)
    if args.do_test and (rank is None or rank == 0):
        test_net(args, config)


if __name__ == '__main__':
    main()
