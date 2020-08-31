#----------------------------------------
#--------- Torch related imports --------
#----------------------------------------
import torch

def optim_SGD(model, initial_lr=0.1, momentum=0.9, weight_decay=0.05):

    # return an SGD optimizer object
    return torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)

def lr_scheduler_StepLR(optimizer, step_size=30, gamma=0.1):

    # return an StepLR object
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

def to_cuda(batch):
    # convert batch: tuple to batch: list
    batch = list(batch)

    for i in range(len(batch)):
        assert isinstance(batch[i], torch.Tensor), "Each element of batch is not a tensor"
        batch[i] = batch[i].cuda(non_blocking=True)

    return batch
