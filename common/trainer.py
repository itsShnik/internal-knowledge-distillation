#----------------------------------------
#--------- Python related imports -------
#----------------------------------------
import os
import time
from collections import namedtuple
import wandb

#----------------------------------------
#--------- Torch related imports --------
#----------------------------------------
import torch
import torch.nn as nn

#----------------------------------------
#--------- Local file imports -----------
#----------------------------------------
from functions.val import do_validation
from common.utils import to_cuda

# Parameter to pass to batch_end_callback
BatchEndParam = namedtuple('BatchEndParams',
                           ['epoch',
                            'nbatch',
                            'rank',
                            'add_step',
                            'data_in_time',
                            'data_transfer_time',
                            'forward_time',
                            'backward_time',
                            'optimizer_time',
                            'metric_time'])

def _multiple_callbacks(callbacks, *args, **kwargs):
    """Sends args and kwargs to any configured callbacks.
    This handles the cases where the 'callbacks' variable
    is ``None``, a single function, or a list.
    """
    if isinstance(callbacks, list):
        for cb in callbacks:
            cb(*args, **kwargs)
        return
    if callbacks:
        callbacks(*args, **kwargs)


def train(config,
        net,
        optimizer,
        train_loader,
        train_metrics,
        val_loader,
        val_metrics,
        criterion=nn.CrossEntropyLoss(),
        rank=None,
        batch_end_callbacks=None,
        epoch_end_callbacks=None):

    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):

        print('PROGRESS: %.2f%%' % (100.0 * epoch / config.TRAIN.END_EPOCH))

        # set the net to train mode
        net.train()

        # reset the train metrics
        train_metrics.reset()

        # initialize end time
        end_time = time.time()

        # start training
        for nbatch, batch in enumerate(train_loader):
            global_steps = len(train_loader) * epoch + nbatch

            # record time
            data_in_time = time.time() - end_time

            # transfer data to GPU
            data_transfer_time = time.time()
            images, labels = to_cuda(batch)
            data_transfer_time = time.time() - data_transfer_time


            # forward time
            forward_time = time.time()
            outputs = net(images)
            forward_time = time.time() - forward_time

            # calculate losses
            loss = criterion(outputs, labels)

            # update training metrics
            metric_time = time.time()
            train_metrics.update(outputs, labels, loss.item())
            metric_time = time.time() - metric_time

            # clear the gradients
            optimizer.zero_grad()

            # backward time
            backward_time = time.time()
            loss.backward()
            backward_time = time.time() - backward_time

            # optimizer time
            optimizer_time = time.time()
            optimizer.step()
            optimizer_time = time.time() - optimizer_time

            # execute batch_end_callbacks
            if batch_end_callbacks is not None:
                batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch, add_step=True, rank=rank,
                                                 data_in_time=data_in_time, data_transfer_time=data_transfer_time,
                                                 forward_time=forward_time, backward_time=backward_time,
                                                 optimizer_time=optimizer_time, metric_time=metric_time
                                                 )
                _multiple_callbacks(batch_end_callbacks, batch_end_params)


            if nbatch % 50 == 0:
                # Print accuracy and loss
                metrics = train_metrics.get()
                wandb.log({'Batch Accuracy': metrics["batch_accuracy"], 'Train Accuracy': metrics['training_accuracy'], 'Train Loss':metrics['training_loss']})
                print('[Rank: {}] [Epoch: {}/{}] [Batch: {}/{}] Batch Accuracy: {:.4f} Training Accuracy: {:.4f} Loss: {}'.format(0 if rank is None else rank, epoch, config.TRAIN.END_EPOCH, nbatch, len(train_loader), metrics["batch_accuracy"],  metrics["training_accuracy"], metrics["training_loss"]))

            # update end time
            end_time = time.time()

        # First do validation at the end of each epoch
        val_acc = do_validation(net, val_loader)

        # update validation metrics
        val_metrics.update(epoch, val_acc)

        # obtain val metrics
        metrics = val_metrics.get()

        # log val metrics
        wandb.log({'Val Acc': metrics['current_val_acc']})

        # print the validation accuracy
        print('Validation accuracy for epoch {}: {:.4f}'.format(epoch, metrics["current_val_acc"]))

        if epoch_end_callbacks is not None:
            _multiple_callbacks(epoch_end_callbacks, epoch, net, optimizer)

