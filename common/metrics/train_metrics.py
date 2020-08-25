#----------------------------------------
#--------- Torch related imports --------
#----------------------------------------
import torch
import torch.distributed as distributed

class TrainMetrics():
    def __init__(self, config, allreduce=False):
        self.correct_instances = torch.tensor(0.)
        self.total_instances = torch.tensor(0.)
        self.training_accuracy = torch.tensor(0.)
        self.batch_loss = torch.tensor(0.)
        self.allreduce = allreduce

    def update(self, outputs, labels, loss):
        # calculate the number of correct instances
        predicted = torch.argmax(outputs.data, 1)
        self.total_instances += labels.size(0)
        self.correct_instances += (predicted == labels).sum().item()

        # share this data across GPUs
        if self.allreduce:
            total_instances = self.total_instances.clone().cuda()
            correct_instances = self.correct_instances.clone().cuda()

            # sum across GPUs
            distributed.all_reduce(total_instances, op=distributed.ReduceOp.SUM)
            distributed.all_reduce(correct_instances, op=distributed.ReduceOp.SUM)

            self.training_accuracy = (correct_instances / total_instances).detach().cpu()

        else:
            self.training_accuracy = (self.correct_instances / self.total_instances).detach().cpu()

        self.batch_loss = loss

    def get():
        return {'training_loss':self.batch_loss, 'training_accuracy':self.training_accuracy}




