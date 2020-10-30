#----------------------------------------
#--------- Torch related imports --------
#----------------------------------------
import torch
import torch.distributed as distributed

#----------------------------------------
#--------- Local Imports ----------------
#----------------------------------------
from common.metrics.basic_metrics import TrainAccuracyMetric, TrainLossMetric

class TrainMetrics():
    def __init__(self, config, allreduce=False):
        self.allreduce = allreduce

    def reset(self):
        # To reset the metrics after every epoch
        # Create a single dict containing all the metrics
        self.all_metrics = {}

    def store(self, metric_name, metric_value, metric_type='Accuracy'):

        if metric_name not in self.all_metrics:
            # initialize a new metric
            self.all_metrics[metric_name] = eval(f'Train{metric_type}Metric')(metric_value, allreduce=self.allreduce)
        else:
            # update the metric
            self.all_metrics[metric_name].update(metric_value)

    def wandb_log(self, step):

        for metric_name, metric in self.all_metrics.items(): 
            metric.wandb_log(metric_name, step)
        
