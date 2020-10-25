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
from common.gumbel_softmax import gumbel_softmax
from common.losses import loss_fn_kd, loss_fn_kd_frozen_teacher

# Define the PolicyVec here
PolicyVec = {
        'SpotTune':12,
        'binary':36
        }

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
        policy_net=None,
        policy_optimizer=None,
        teacher_net=None,
        rank=None,
        batch_end_callbacks=None,
        epoch_end_callbacks=None):

    for epoch in range(config.TRAIN.BEGIN_EPOCH, config.TRAIN.END_EPOCH):

        # We need to visualize policy vectors
        if config.NETWORK.TRAINING_STRATEGY in PolicyVec:
            policy_decisions = torch.zeros(PolicyVec[config.NETWORK.TRAINING_STRATEGY]).cuda(non_blocking=True)
            policy_max = 0
        else:
            policy_decisions = None
            policy_max = None

        print('PROGRESS: %.2f%%' % (100.0 * epoch / config.TRAIN.END_EPOCH))

        # set the net to train mode
        net.train()

        # policy net to train
        if config.NETWORK.TRAINING_STRATEGY in PolicyVec:
            policy_net.train()

        # teacher net to eval
        if config.NETWORK.TRAINING_STRATEGY == 'knowledge_distillation':
            teacher_net.eval()

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

            # check for policy net
            if config.NETWORK.TRAINING_STRATEGY in PolicyVec:
                policy_vector = policy_net(images)
                policy_action = gumbel_softmax(policy_vector.view(policy_vector.size(0), -1, 2))
                policy = policy_action[:,:,1]
                policy_decisions = policy_decisions + policy.clone().detach().sum(0)
                policy_max += policy.size(0)
                outputs = net(images, policy)
                loss = criterion(outputs, labels)

            elif config.NETWORK.TRAINING_STRATEGY == 'AdditionalHeads':
                outputs, additional_outputs = net(images)
                loss = criterion(outputs, labels)
                for additional_output in additional_outputs:
                    additional_loss = criterion(additional_output, labels)

                    if config.USE_KD_LOSS:
                        alpha = config.ALPHA
                        temp = config.TEMPERATURE

                        loss_kd = eval(config.KD_LOSS_FUNCTION)(additional_output, outputs, alpha, temp)
                        additional_loss = loss_kd + additional_loss * (1-alpha)

                    loss += additional_loss

            elif config.NETWORK.TRAINING_STRATEGY == 'knowledge_distillation':
                teacher_outputs = teacher_net(images)
                outputs = net(images)

                # combine both the losses using params, that come along with the teacher
                alpha = config.TEACHER.ALPHA
                temp = config.TEACHER.TEMPERATURE

                loss_kd = eval(config.TEACHER.KD_LOSS_FUNCTION)(outputs, teacher_outputs, alpha, temp)
                loss_cls = criterion(outputs, labels)

                loss = loss_kd + loss_cls * (1-alpha)

            else:
                outputs = net(images)
                loss = criterion(outputs, labels)

            forward_time = time.time() - forward_time

            # update training metrics
            metric_time = time.time()
            train_metrics.update(outputs, labels, loss.item())
            metric_time = time.time() - metric_time

            # clear the gradients
            optimizer.zero_grad()
            if config.NETWORK.TRAINING_STRATEGY in PolicyVec:
                policy_optimizer.zero_grad()

            # backward time
            backward_time = time.time()
            loss.backward()
            backward_time = time.time() - backward_time

            # optimizer time
            optimizer_time = time.time()
            optimizer.step()
            if config.NETWORK.TRAINING_STRATEGY in PolicyVec:
                policy_optimizer.step()
            optimizer_time = time.time() - optimizer_time

            # Log the optimizer stats -- LR
            for i, param_group in enumerate(optimizer.param_groups):
                wandb.log({f'LR_{i}': param_group['lr']})

            # Log the optim stats for Policy Optim
            if config.NETWORK.TRAINING_STRATEGY in PolicyVec:
                for i, param_group in enumerate(policy_optimizer.param_groups):
                    wandb.log({f'Policy_LR_{i}': param_group['lr']})

            # execute batch_end_callbacks
            if batch_end_callbacks is not None:
                batch_end_params = BatchEndParam(epoch=epoch, nbatch=nbatch, add_step=True, rank=rank,
                                                 data_in_time=data_in_time, data_transfer_time=data_transfer_time,
                                                 forward_time=forward_time, backward_time=backward_time,
                                                 optimizer_time=optimizer_time, metric_time=metric_time
                                                 )
                _multiple_callbacks(batch_end_callbacks, batch_end_params)


            if nbatch % 100 == 0:
                # Print accuracy and loss
                metrics = train_metrics.get()
                wandb.log({'Batch Accuracy': metrics["batch_accuracy"], 'Train Accuracy': metrics['training_accuracy'], 'Train Loss':metrics['training_loss']})
                print('[Rank: {}] [Epoch: {}/{}] [Batch: {}/{}] Batch Accuracy: {:.4f} Training Accuracy: {:.4f} Loss: {}'.format(0 if rank is None else rank, epoch, config.TRAIN.END_EPOCH, nbatch, len(train_loader), metrics["batch_accuracy"],  metrics["training_accuracy"], metrics["training_loss"]))

            # update end time
            end_time = time.time()

        # First do validation at the end of each epoch
        if config.NETWORK.TRAINING_STRATEGY == 'AdditionalHeads':
            val_acc, additional_val_acc = do_validation(config, net, val_loader, policy_net=policy_net)
        else:
            val_acc = do_validation(config, net, val_loader, policy_net=policy_net)

        # update validation metrics
        val_metrics.update(epoch, val_acc)

        # obtain val metrics
        metrics = val_metrics.get()

        # log val metrics
        if config.NETWORK.TRAINING_STRATEGY == 'AdditionalHeads':
            for index, acc in enumerate(additional_val_acc):
                wandb.log({f'Additional Val Acc {index+1}':acc})

        wandb.log({'Val Acc': metrics['current_val_acc'], 'Best Val Acc': metrics['best_val_acc'], 'Best Val Epoch': metrics['best_val_epoch']})

        # print the validation accuracy
        print('Validation accuracy for epoch {}: {:.4f}'.format(epoch, metrics["current_val_acc"]))

        if epoch_end_callbacks is not None:
            _multiple_callbacks(epoch_end_callbacks, rank=rank if rank is not None else 0, epoch=epoch, net=net, optimizer=optimizer, policy_net=policy_net, policy_optimizer=policy_optimizer, policy_decisions=policy_decisions, policy_max=policy_max, training_strategy=config.NETWORK.TRAINING_STRATEGY)

