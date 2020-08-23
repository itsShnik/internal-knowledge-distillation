#----------------------------------------
#--------- Python related imports -------
#----------------------------------------
import os
import time
from collections import namedtuple

#----------------------------------------
#--------- Torch related imports
#----------------------------------------
import torch

def to_cuda(batch):
    # convert batch: tuple to batch: list
    batch = list(batch)

    for i in range(len(batch)):
        assert isinstance(batch[i], torch.Tensor), "Each element of batch is not a tensor"
            batch[i] = batch[i].cuda(non_blocking=True)

    return batch

def train(config,
        net,
        optimizer,
        lr_scheduler=None,
        rank=None):

    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):

        print('PROGRESS: %.2f%%' % (100.0 * epoch / config.END_EPOCH))

        # set the net to train mode
        net.train()

        # initialize end time
        end_time = time.time()

        # start training
        for nbatch, batch in enumerate(train_loader):
            global_steps = len(train_loader) * epoch + nbatch

            # record time
            data_in_time = time.time() - end_time

            # transfer data to GPU
            data_transfer_time = time.time()
            batch = to_cuda(batch)
            data_transfer_time = time.time() - data_transfer_time
            # clear the gradients
            optimizer.zero_grad()

            # forward time
            forward_time = time.time()

            # forward pass
            outputs, loss = net(*batch)

            # forward_time
            forward_time = time.time() - forward_time

            # backward time
            backward_time = time.time()
            loss.backward()
            backward_time = time.time() - backward_time

            # optimizer time
            optimizer_time = time.time()
            
            if lr_scheduler is not None:
                lr_scheduler.step()

            optimizer.step()
            optimizer_time = time.time() - optimizer_time

            #TODO: batch end callbacks, once configured

            # update end time
            end_time = time.time()

        #TODO: execute epoch_end_callbacks, once configured
