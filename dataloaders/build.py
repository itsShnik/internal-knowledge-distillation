#----------------------------------------
#--------- Torch related imports --------
#----------------------------------------
import torch

#----------------------------------------
#--------- Funcs and Classes for Datasets
#----------------------------------------
from datasets.cubs import CUBS
from datasets.decathlon import decathlon_dataset

DATASETS = {'cubs':CUBS}
DECATHLON_DATASETS = {'imagenet12', 'cifar100', 'aircraft', 'gtsrb', 'omniglot', 'dtd', 'vgg-flowers', 'daimlerpedcls', 'svhn', 'ucf101'}

def build_dataset(dataset_name, *args, **kwargs):
    assert dataset_name in DATASET_CATALOGS, "dataset not in catalogs"
    return DATASET_CATALOGS[dataset_name](*args, **kwargs)

def make_data_sampler(dataset, shuffle, distributed, num_replicas, rank):
    if distributed:
        return torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle, num_replicas=num_replicas, rank=rank)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler

def make_batch_data_sampler(dataset, sampler, batch_size):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, batch_size, drop_last=False
    )
    return batch_sampler

def make_dataloader(config, dataset=None, mode='train', distributed=False, num_replicas=None, rank=None):

    # config variables
    num_gpu = len(config.GPUS.split(','))
    num_workers = config.NUM_WORKERS_PER_GPU * num_gpu

    if mode == 'train':
        aspect_grouping = config.TRAIN.ASPECT_GROUPING
        batch_size = config.TRAIN.BATCH_IMAGES * num_gpu
        shuffle = config.TRAIN.SHUFFLE
        splits = config.DATASET.TRAIN_SPLIT
    else:
        aspect_grouping = False
        batch_size = config.VAL.BATCH_IMAGES * num_gpu
        shuffle = config.VAL.SHUFFLE
        splits = config.DATASET.VAL_SPLIT

    # create a Dataset class object
    if dataset is None:
        if config.DATASET.DATASET_NAME in DECATHLON_DATASETS:
            dataset = decathlon_dataset(name=config.DATASET.DATASET_NAME, root=config.DATASET.ROOT_PATH, splits=splits, toy=config.DATASET.TOY)
        else:
            dataset = build_dataset(dataset_name=config.DATASET.DATASET, root=config.DATASET.ROOT_PATH, splits=splits, toy=config.DATASET.TOY)

    sampler = make_data_sampler(dataset, shuffle, distributed, num_replicas, rank)
    batch_sampler = make_batch_data_sampler(dataset, sampler, batch_size)

    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_sampler=batch_sampler,
                                             num_workers=num_workers,
                                             pin_memory=False)

    return dataloader
