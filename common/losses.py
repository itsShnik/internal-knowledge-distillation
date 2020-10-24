#----------------------------------------
#--------- Torch related imports --------
#----------------------------------------
import torch.nn as nn
import torch.nn.functional as F

def loss_fn_kd(student_outputs, teacher_outputs, alpha=0.5, temp=10):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha

    NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
    and student expects the input tensor to be log probabilities! See Issue #2
    """

    kd_loss = nn.KLDivLoss()(F.log_softmax(student_outputs/temp, dim=1),
                            F.softmax(teacher_outputs/temp, dim=1)) * (alpha * temp * temp)

    return kd_loss
