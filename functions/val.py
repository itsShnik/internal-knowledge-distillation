#----------------------------------------
#--------- Torch related imports --------
#----------------------------------------
import torch

#----------------------------------------
#--------- Common and utils imports -----
#----------------------------------------
from common.utils.misc import to_cuda
from common.utils.gumbel_softmax import gumbel_softmax

@torch.no_grad()
def do_validation(config, net, val_loader, policy_net=None):
    net.eval()
    if policy_net is not None:
        policy_net.eval()
    correct_instances = 0
    total_instances = 0

    if config.NETWORK.TRAINING_STRATEGY in ['AdditionalHeads', 'AdditionalStochastic']:
        additional_correct_instances = [0] * config.NETWORK.NUM_ADDITIONAL_HEADS
        additional_total_instances = [0] * config.NETWORK.NUM_ADDITIONAL_HEADS

    for nbatch, batch in enumerate(val_loader):
        images, labels = to_cuda(batch)

        # Forward pass
        if policy_net is not None:
            policy_vector = policy_net(images)
            policy_action = gumbel_softmax(policy_vector.view(policy_vector.size(0), -1, 2))
            policy = policy_action[:,:,1]
            outputs = net(images, policy)
        elif config.NETWORK.TRAINING_STRATEGY in ['AdditionalHeads', 'AdditionalStochastic']:
            outputs, additional_outputs = net(images, mode='val')
        else:
            outputs = net(images, mode='val')

        # calculate the accuracy
        predicted = torch.argmax(outputs.data, 1)
        total_instances += labels.size(0)
        correct_instances += (predicted == labels).sum().item()

        # calculate additional accuracy if have to 
        if config.NETWORK.TRAINING_STRATEGY in ['AdditionalHeads', 'AdditionalStochastic']:
            for i in range(config.NETWORK.NUM_ADDITIONAL_HEADS):
                additional_predicted = torch.argmax(additional_outputs[i].data, 1)
                additional_total_instances[i] += labels.size(0)
                additional_correct_instances[i] += (additional_predicted == labels).sum().item()

    # return accuracy
    val_acc = (100.0 * correct_instances) / total_instances

    # additional val acc
    if config.NETWORK.TRAINING_STRATEGY in ['AdditionalHeads', 'AdditionalStochastic']:
        return_list = []
        for i in range(config.NETWORK.NUM_ADDITIONAL_HEADS):
            additional_val_acc = (100.0 * additional_correct_instances[i]) / additional_total_instances[i]
            return_list.append(additional_val_acc)
        return val_acc, return_list

    return val_acc
