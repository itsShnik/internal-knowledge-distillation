#----------------------------------------
#--------- Torch related imports --------
#----------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

def calculate_loss_and_accuracy(criterion, outputs, labels):

    loss = criterion(outputs, labels)

    # calculate the accuracy here
    predicted = torch.argmax(outputs.data, 1)
    correct_instances = (predicted == labels).sum().item() 
    total_instances = labels.size(0)
    accuracy = 100.0 * (correct_instances / total_instances)

    return loss, accuracy

def loss_fn_kd(student_outputs, teacher_outputs, alpha, temp):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """

    kd_loss = nn.KLDivLoss()(F.log_softmax(student_outputs/temp, dim=1),
                            F.softmax(teacher_outputs/temp, dim=1))

    return kd_loss

def loss_fn_kd_frozen_teacher(student_outputs, teacher_outputs, alpha, temp):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """
    teacher_outputs_clone = teacher_outputs.clone().detach()

    kd_loss = nn.KLDivLoss()(F.log_softmax(student_outputs/temp, dim=1),
                            F.softmax(teacher_outputs_clone/temp, dim=1))

    return kd_loss
