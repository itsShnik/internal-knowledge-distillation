#----------------------------------------
#--------- Torch related imports --------
#----------------------------------------
import torch
from common.utils import to_cuda

@torch.no_grad()
def do_validation(net, val_loader):
    net.eval()
    correct_instances = 0
    total_instances = 0
    for nbatch, batch in enumerate(val_loader):
        images, labels = to_cuda(batch)

        # Forward pass
        outputs = net(images)

        # calculate the accuracy
        predicted = torch.argmax(outputs.data, 1)
        total_instances += labels.size(0)
        correct_instances += (predicted == labels).sum().item()

    # return accuracy
    return (100.0 * correct_instances) / total_instances
