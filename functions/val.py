#----------------------------------------
#--------- Torch related imports --------
#----------------------------------------
import torch

#----------------------------------------
#--------- Common and utils imports -----
#----------------------------------------
from common.utils import to_cuda
from common.gumbel_softmax import gumbel_softmax

@torch.no_grad()
def do_validation(config, net, val_loader, policy_net=None):
    net.eval()
    if policy_net is not None:
        policy_net.eval()
    correct_instances = 0
    total_instances = 0

    if config.NETWORK.TRAINING_STRATEGY == 'AdditionalHeads':
        additional_correct_instances = 0
        additional_total_instances = 0

    for nbatch, batch in enumerate(val_loader):
        images, labels = to_cuda(batch)

        # Forward pass
        if policy_net is not None:
            policy_vector = policy_net(images)
            policy_action = gumbel_softmax(policy_vector.view(policy_vector.size(0), -1, 2))
            policy = policy_action[:,:,1]
            outputs = net(images, policy)
        elif config.NETWORK.TRAINING_STRATEGY == 'AdditionalHeads':
            outputs, additional_outputs = net(images, config.NETWORK.ADDITIONAL_MASKS)
        else:
            outputs = net(images)

        # calculate the accuracy
        predicted = torch.argmax(outputs.data, 1)
        total_instances += labels.size(0)
        correct_instances += (predicted == labels).sum().item()

        # calculate additional accuracy if have to 
        if config.NETWORK.TRAINING_STRATEGY == 'AdditionalHeads':
            additional_predicted = torch.argmax(additional_outputs.data, 1)
            additional_total_instances += labels.size(0)
            additional_correct_instances += (additional_predicted == labels).sum().item()

    # return accuracy
    val_acc = (100.0 * correct_instances) / total_instances

    # additional val acc
    if config.NETWORK.TRAINING_STRATEGY == 'AdditionalHeads':
        additional_val_acc = (100.0 * additional_correct_instances) / additional_total_instances
        return val_acc, additional_val_acc

    return val_acc
