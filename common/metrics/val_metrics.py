#----------------------------------------
#--------- Torch Related Imports --------
#----------------------------------------
import torch
import torch.distributed as distributed

class ValMetrics():
    def __init__(self, config, allreduce=False):
        self.current_epoch = 0
        self.best_val_epoch = 0
        self.current_val_acc = 0.
        self.best_val_acc = 0.
        self.updated_best_val = True

        self.allreduce = allreduce

    def update(self, epoch, val_acc):
        self.current_epoch = epoch

        # convert val_acc to tensor
        # val_acc = torch.tensor(val_acc)

        #if self.allreduce:
        #    val_acc = val_acc.clone().cuda()
        #    gpus = torch.tensor(1.).cuda()

        #    # Sum validation accuracies across GPUs
        #    distributed.all_reduce(val_acc, op=distributed.ReduceOp.SUM)
        #    distributed.all_reduce(gpus, op=distributed.ReduceOp.SUM)

        #    val_acc = (val_acc / gpus).cpu()


        # current validation accuracy
        self.current_val_acc = val_acc

        # compare with best val acc
        if self.current_val_acc > self.best_val_acc:
            self.best_val_acc = self.current_val_acc
            self.best_val_epoch = epoch
            self.updated_best_val = True
        else:
            self.updated_best_val = False

    def get(self):
        return {'current_val_acc':self.current_val_acc, 'best_val_acc':self.best_val_acc, 'best_val_epoch':self.best_val_epoch, 'updated_best_val':self.updated_best_val}


