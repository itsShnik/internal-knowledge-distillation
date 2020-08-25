class Checkpoint():
    def __init__(self, config, val_metrics):
        self.config = config
        self.val_metrics = val_metrics

        self.save_path = self.set_up_logging_dir()

    def __call__(self, epoch, net, optimizer):
        if val_metrics.updated_best_val:
