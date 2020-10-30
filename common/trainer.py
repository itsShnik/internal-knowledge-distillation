#----------------------------------------
#--------- Python related imports -------
#----------------------------------------
import os
import time
from collections import namedtuple
import wandb
from prettytable import PrettyTable

#----------------------------------------
#--------- Torch related imports --------
#----------------------------------------
import torch
import torch.nn as nn

#----------------------------------------
#--------- Local file imports -----------
#----------------------------------------
from functions.val import do_validation
from common.utils.misc import to_cuda
from common.utils.gumbel_softmax import gumbel_softmax
from common.utils.losses import loss_fn_kd, loss_fn_kd_frozen_teacher, calculate_loss_and_accuracy

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
                loss, accuracy = calculate_loss_and_accuracy(criterion, outputs, labels)

                # store the obtained metrics
                train_metrics.store('training_loss', loss.item(), 'Loss')
                train_metrics.store('training_accuracy', accuracy, 'Accuracy')

            elif config.NETWORK.TRAINING_STRATEGY == 'AdditionalHeads':
                outputs, additional_outputs = net(images)
                loss, accuracy = calculate_loss_and_accuracy(criterion, outputs, labels)

                # Store the metrics obtained until now
                train_metrics.store('training_loss', loss.item(), 'Loss')
                train_metrics.store('training_accuracy', accuracy, 'Accuracy')

                for i, additional_output in enumerate(additional_outputs):
                    additional_loss, additional_accuracy = calculate_loss_and_accuracy(criterion, additional_output, labels)

                    # log the additional branch classification loss and training acc
                    train_metrics.store(f'additional_training_loss_{i+1}', additional_loss.item(), 'Loss')
                    train_metrics.store(f'additional_training_accuracy_{i+1}', additional_accuracy, 'Accuracy')

                    if config.USE_KD_LOSS:
                        alpha = config.ALPHA
                        temp = config.TEMPERATURE

                        loss_kd = eval(config.KD_LOSS_FUNCTION)(additional_output, outputs, alpha, temp)
                        additional_loss = loss_kd * (alpha * temp * temp) + additional_loss * (1-alpha)

                        # Store the KLDiv between the main and this branch
                        train_metrics.store(f'kl_div_with_branch_{i+1}', loss_kd.item(), 'Loss')


                    loss += additional_loss

            elif config.NETWORK.TRAINING_STRATEGY == 'knowledge_distillation':
                teacher_outputs = teacher_net(images)
                outputs = net(images)

                # combine both the losses using params, that come along with the teacher
                alpha = config.TEACHER.ALPHA
                temp = config.TEACHER.TEMPERATURE

                loss_kd = eval(config.TEACHER.KD_LOSS_FUNCTION)(outputs, teacher_outputs, alpha, temp)
                loss_cls, accuracy = calculate_loss_and_accuracy(criterion, outputs, labels)

                train_metrics.store(f'kl_div_student_teahcer', loss_kd.item(), 'Loss')
                train_metrics.store(f'training_loss', loss_cls.item(), 'Loss')
                train_metrics.store(f'training_accuracy', accuracy, 'Accuracy')

                loss = loss_kd * (alpha * temp * temp) + loss_cls * (1-alpha)

            else:
                outputs = net(images)
                loss, accuracy = calculate_loss_and_accuracy(criterion, outputs, labels)

                # store the obtained metrics
                train_metrics.store('training_loss', loss.item(), 'Loss')
                train_metrics.store('training_accuracy', accuracy, 'Accuracy')

            forward_time = time.time() - forward_time

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
                print('\n---------------------------------')
                print(f'[Rank: {rank if rank is not None else 0}], [Epoch: {epoch}/{config.TRAIN.END_EPOCH}], [Batch: {nbatch}/{len(train_loader)}]')
                print('-----------------------------------\n')

                table = PrettyTable(['Metric', 'Value'])
                for metric_name, metric in train_metrics.all_metrics.items():
                    table.add_row([metric_name, metric.current_value])
                print(table)

            # update end time
            end_time = time.time()

        # First do validation at the end of each epoch
        if config.NETWORK.TRAINING_STRATEGY == 'AdditionalHeads':
            val_acc, additional_val_acc = do_validation(config, net, val_loader, policy_net=policy_net)
            # update validation metrics
            val_metrics.store('val_accuracy', val_acc, 'Accuracy')
            for i, acc in enumerate(additional_val_acc):
                val_metrics.store(f'additional_val_accuracy_{i+1}', acc, 'Accuracy')
        else:
            val_acc = do_validation(config, net, val_loader, policy_net=policy_net)
            # update validation metrics
            val_metrics.store('val_accuracy', val_acc, 'Accuracy')

        # Log the optimizer stats -- LR
            for i, param_group in enumerate(optimizer.param_groups):
                wandb.log({f'LR_{i}': param_group['lr']}, step=epoch)

            # Log the optim stats for Policy Optim
            if config.NETWORK.TRAINING_STRATEGY in PolicyVec:
                for i, param_group in enumerate(policy_optimizer.param_groups):
                    wandb.log({f'Policy_LR_{i}': param_group['lr']}, step=epoch)

        # Log both the training and validation metrics
        train_metrics.wandb_log(epoch)
        val_metrics.wandb_log(epoch)

        # print the validation accuracy
        print('\n-----------------')
        print('Validation Metrics')
        print('-----------------\n')

        table = PrettyTable(['Metric', 'Current Value', 'Best Value'])
        for metric_name, metric in val_metrics.all_metrics.items():
            table.add_row([metric_name, metric.current_value, metric.best_value if 'best_value' in dir(metric) else '----'])
        print(table)

        if epoch_end_callbacks is not None:
            _multiple_callbacks(epoch_end_callbacks, rank=rank if rank is not None else 0, epoch=epoch, net=net, optimizer=optimizer, policy_net=policy_net, policy_optimizer=policy_optimizer, policy_decisions=policy_decisions, policy_max=policy_max, training_strategy=config.NETWORK.TRAINING_STRATEGY)

