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
def do_validation(net, val_loader, policy_net=None):
    net.eval()
    if policy_net is not None:
        policy_net.eval()
    correct_instances = 0
    total_instances = 0
    for nbatch, batch in enumerate(val_loader):
        images, labels = to_cuda(batch)

        # Forward pass
        if policy_net is not None:
            policy_vector = policy_net(images)
            policy_action = gumbel_softmax(policy_vector.view(policy_vector.size(0), -1, 2))
            policy = policy_action[:,:,1]
            outputs = net(images, policy)
        else:
            outputs = net(images)

        # calculate the accuracy
        predicted = torch.argmax(outputs.data, 1)
        total_instances += labels.size(0)
        correct_instances += (predicted == labels).sum().item()

    # return accuracy
    val_acc = (100.0 * correct_instances) / total_instances
    return val_acc
