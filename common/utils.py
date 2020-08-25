#----------------------------------------
#--------- Torch related imports --------
#----------------------------------------
import torch

def optim_SGD(model, initial_lr=0.1, momentum=0.9, weight_decay=0.05):

    # return an SGD optimizer object
    return torch.optim.SGD(filter(lambda f: f.requires_grad, model.parameters()), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)

def lr_scheduler_StepLR(optimizer, step_size=30, gamma=0.1):

    # return an StepLR object
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
