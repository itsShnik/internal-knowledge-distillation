#----------------------------------------
#--------- Torch related imports --------
#----------------------------------------
import torch
import torch.distributed as distributed

class TrainMetrics():
    def __init__(self, config, allreduce=False):
        self.allreduce = allreduce
        self.reset()

    def reset(self):
        # To reseet the metrics after every epoch
        self.correct_instances = 0
        self.total_instances = 0
        self.average_training_accuracy = 0.
        self.batch_accuracy = 0.
        self.batch_loss = 0.

    def update(self, outputs, labels, loss):
        # calculate the number of correct instances
        predicted = torch.argmax(outputs.data, 1)
        new_correct_instances = (predicted == labels).sum().cpu().item()
        new_instances = labels.size(0)

        # update total instances
        self.total_instances += new_instances
        self.correct_instances += new_correct_instances

        # share this data across GPUs
        if self.allreduce:
            total_instances = torch.tensor(self.total_instances).clone().cuda()
            correct_instances = torch.tensor(self.correct_instances).clone().cuda()

            # sum across GPUs
            distributed.all_reduce(total_instances, op=distributed.ReduceOp.SUM)
            distributed.all_reduce(correct_instances, op=distributed.ReduceOp.SUM)

            self.average_training_accuracy = 100.0 * (correct_instances.cpu().item() / total_instances.cpu().item())

        else:
            self.average_training_accuracy = 100.0 * (self.correct_instances / self.total_instances)


        self.batch_accuracy = 100.0 * (new_correct_instances / new_instances)
        self.batch_loss = loss

    def get(self):
        return {'training_loss':self.batch_loss, 'batch_accuracy':self.batch_accuracy, 'training_accuracy':self.average_training_accuracy}




