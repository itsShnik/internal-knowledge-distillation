#----------------------------------------
#--------- OS related imports -----------
#----------------------------------------
import os
import wandb

#----------------------------------------
#--------- Torch related imports --------
#----------------------------------------
import numpy as np
import torch
import torch.nn
import torch.optim as optim
import torch.distributed as distributed
import torchvision.models as models
from torch.nn.parallel import DistributedDataParallel as DDP
from torchsummary import summary

#----------------------------------------
#--------- Model related imports --------
#----------------------------------------
from modules.networks_for_modified_resnet import modified_resnet, modified_dynamic_resnet
from modules.networks_for_original_resnet import resnet18, resnet34, resnet50

#----------------------------------------
#--------- Dataloader related imports ---
#----------------------------------------
from dataloaders.build import make_dataloader, build_dataset

#----------------------------------------
#--------- Imports from common ----------
#----------------------------------------
from common.optim import optim_SGD, optim_AdamW
from common.utils import smart_model_load
from common.trainer import PolicyVec, train
from common.metrics.train_metrics import TrainMetrics
from common.metrics.val_metrics import ValMetrics
from common.callbacks.epoch_end_callbacks.checkpoint import Checkpoint
from common.callbacks.epoch_end_callbacks.lrscheduler import LRScheduler, LRSchedulerPolicy
from common.callbacks.epoch_end_callbacks.visualization_plotter import VisualizationPlotter

def train_net(args, config):

    # manually set random seed
    if config.RNG_SEED > -1:
        np.random.seed(config.RNG_SEED)
        torch.random.manual_seed(config.RNG_SEED)
        torch.cuda.manual_seed_all(config.RNG_SEED)

    # cudnn
    torch.backends.cudnn.benchmark = False
    if args.cudnn_off:
        torch.backends.cudnn.enabled = False

    # parallel: distributed training for utilising multiple GPUs
    if args.dist:
        # set up the environment
        local_rank = int(os.environ.get('LOCAL_RANK') or 0)
        config.GPUS = str(local_rank)
        torch.cuda.set_device(local_rank)
        master_address = os.environ['MASTER_ADDR']
        master_port = int(os.environ['MASTER_PORT'] or 23456)
        world_size = int(os.environ['WORLD_SIZE'] or 1)
        rank = int(os.environ['RANK'] or 0)

        # initialize process group
        distributed.init_process_group(
            backend='nccl',
            init_method='tcp://{}:{}'.format(master_address, master_port),
            world_size=world_size,
            rank=rank,
            group_name='mtorch'
        )
        print(f'native distributed, size: {world_size}, rank: {rank}, local rank: {local_rank}')

        # set cuda devices
        torch.cuda.set_device(local_rank)
        config.GPUS = str(local_rank)

        # initialize the model and put it to GPU
        model = eval(config.MODULE)(config=config.NETWORK)
        model = model.cuda()

        # wrap the model using torch distributed data parallel
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        # Check if the model requires policy network
        if config.NETWORK.TRAINING_STRATEGY in PolicyVec:
            policy_model = eval(config.POLICY_MODULE)(config=config.POLICY.NETWORK)
            policy_model = policy_model.cuda()

            # wrap in DDP
            policy_model = DDP(policy_model, device_ids=[local_rank], output_device=local_rank)

        # summarize the model
        if rank == 0:
            print("summarizing the main network")
            summary(model, (3, 224, 224))

            if config.NETWORK.TRAINING_STRATEGY in PolicyVec:
                print("summarizing the policy network")
                summary(policy_model, (3, 64, 64))

        # dataloaders for training, val and test set
        train_loader = make_dataloader(config, mode='train', distributed=True, num_replicas=world_size, rank=rank)
        val_loader = make_dataloader(config, mode='val', distributed=True, num_replicas=world_size, rank=rank)

    else:
        # single GPU training
        # set CUDA device in env variables
        config.GPUS = str(0)
        torch.cuda.set_device(0)

        # initialize the model and put is to GPU
        model = eval(config.MODULE)(config=config.NETWORK)
        model = model.cuda()

        # check for policy model
        if config.NETWORK.TRAINING_STRATEGY in PolicyVec:
            policy_model= eval(config.POLICY_MODULE)(config=config.POLICY.NETWORK)
            policy_model = policy_model.cuda()

        # summarize the model
        print("summarizing the model")
        print(model)
        summary(model, (3, 64, 64))

        if config.NETWORK.TRAINING_STRATEGY in PolicyVec:
            print("Summarizing the policy model")
            summary(policy_model, (3, 64, 64))

        # dataloaders for training and test set
        train_loader = make_dataloader(config, mode='train', distributed=False)
        val_loader = make_dataloader(config, mode='val', distributed=False)

    # wandb logging
    wandb.watch(model, log='all')
    if config.NETWORK.TRAINING_STRATEGY in PolicyVec:
        wandb.watch(policy_model, log='all')

    # set up the initial learning rate
    initial_lr = config.TRAIN.LR

    # configure the optimizer
    try:
        optimizer = eval(f'optim_{config.TRAIN.OPTIMIZER}')(model=model, initial_lr=initial_lr, momentum=config.TRAIN.MOMENTUM, weight_decay=config.TRAIN.WEIGHT_DECAY)
    except:
        raise ValueError(f'{config.TRAIN.OPTIMIZER}, not supported!!')

    if config.NETWORK.TRAINING_STRATEGY in PolicyVec:
        initial_lr_policy = config.POLICY.LR
        try:
            policy_optimizer = eval(f'optim_{config.POLICY.OPTIMIZER}')(model=model, initial_lr=initial_lr_policy, momentum=config.POLICY.MOMENTUM, weight_decay=config.POLICY.WEIGHT_DECAY)
        except:
            raise ValueError(f'{config.POLICY.OPTIMIZER}, not supported!!')

    # Load pre-trained model
    if config.NETWORK.PRETRAINED_MODEL != '':
        print(f"Loading the pretrained model from {config.NETWORK.PRETRAINED_MODEL} ...")
        pretrain_state_dict = torch.load(config.NETWORK.PRETRAINED_MODEL, map_location = lambda storage, loc: storage)['net_state_dict']
        smart_model_load(model, pretrain_state_dict, loading_method=config.NETWORK.PRETRAINED_LOADING_METHOD)

    # Set up the metrics
    train_metrics = TrainMetrics(config, allreduce=False)
    val_metrics = ValMetrics(config, allreduce=args.dist)

    # Set up the callbacks
    # batch end callbacks
    batch_end_callbacks = None

    # epoch end callbacks
    epoch_end_callbacks = [Checkpoint(config, val_metrics), LRScheduler(config)]
    if config.NETWORK.TRAINING_STRATEGY in PolicyVec:
        epoch_end_callbacks.append(LRSchedulerPolicy(config))
        epoch_end_callbacks.append(VisualizationPlotter())

    # At last call the training function from trainer
    train(config=config, net=model, optimizer=optimizer, train_loader=train_loader, train_metrics=train_metrics, val_loader=val_loader, val_metrics=val_metrics, policy_net=policy_model if config.NETWORK.TRAINING_STRATEGY in PolicyVec else None, policy_optimizer=policy_optimizer if config.NETWORK.TRAINING_STRATEGY in PolicyVec else None, rank=rank if args.dist else None, batch_end_callbacks=batch_end_callbacks, epoch_end_callbacks=epoch_end_callbacks)
