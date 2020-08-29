#----------------------------------------
#--------- Torch related imports --------
#----------------------------------------
import numpy as np
import torch
import torch.nn
import torch.optim as optim
import torch.distributed as distributed
from torch.nn.parallel import DistributedDataParallel as DDP

#----------------------------------------
#--------- Model related imports --------
#----------------------------------------
from modules.resnet26 import resnet26
from policy_modules.resnet8 import resnet8

#----------------------------------------
#--------- Dataloader related imports ---
#----------------------------------------
from dataloaders.build import make_dataloader, build_dataset

#----------------------------------------
#--------- Imports from common ----------
#----------------------------------------
from common.utils import optim_SGD
from common.utils import lr_scheduler_StepLR
from common.trainer import train
from common.metrics.train_metrics import TrainMetrics
from common.metrics.val_metrics import ValMetrics
from common.callbacks.epoch_end_callbacks.checkpoint import Checkpoint

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
        model = eval(config.MODULE)(num_class=config.NUM_CLASSES)
        model = model.cuda()

        # dataloaders for training, val and test set
        train_loader = make_dataloader(config, mode='train', distributed=True, num_replicas=world_size, rank=rank)
        val_loader = make_dataloader(config, mode='val', distributed=True, num_replicas=world_size, rank=rank)

        # set the batch_size
        batch_size = world_size * config.TRAIN.BATCH_IMAGES

    else:
        # single GPU training
        # set CUDA device in env variables
        config.GPUS = str(0)
        torch.cuda.set_device(0)

        # initialize the model and put is to GPU
        model = eval(config.MODULE)(num_class=config.NUM_CLASSES)
        model = model.cuda()

        # dataloaders for training and test set
        train_loader = make_dataloader(config, mode='train', distributed=False)
        val_loader = make_dataloader(config, mode='val', distributed=False)

        # set the batch size
        batch_size = config.TRAIN.BATCH_IMAGES

    # set up the initial learning rate, proportional to batch_size
    initial_lr = batch_size * config.TRAIN.LR

    # configure the optimizer
    try:
        optimizer = eval(f'optim_{config.TRAIN.OPTIMIZER}')(model, initial_lr, config.TRAIN.MOMENTUM, config.TRAIN.WEIGHT_DECAY)
    except:
        raise ValueError(f'{config.TRAIN.OPTIMIZER}, not supported!!')

    # config the learning rates and schedulars
    try:
        lr_scheduler = eval(f'lr_scheduler_{config.TRAIN.LR_SCHEDULER}')(optimizer, step_size=config.TRAIN.STEP_SIZE, gamma=config.TRAIN.GAMMA)
    except:
        raise ValueError(f'{config.TRAIN.LR_SCHEDULER}, not supported!!')

    # Set up the metrics
    train_metrics = TrainMetrics(config, allreduce=args.dist)
    val_metrics = ValMetrics(config, allreduce=args.dist)

    # Set up the callbacks
    # batch end callbacks
    batch_end_callbacks = None

    # epoch end callbacks
    epoch_end_callbacks = [Checkpoint(config), val_metrics]

    #TODO: Broadcast the parameters and optimizer state from rank 0 before the start of training

    # At last call the training function from trainer
    train(config=config, net=model, optimizer=optimizer, train_loader=train_loader, train_metrics=train_metrics, val_loader=val_loader, val_metrics=val_metrics, lr_scheduler=lr_scheduler, rank=rank, batch_end_callbacks=batch_end_callbacks, epoch_end_callbacks=epoch_end_callbacks)
