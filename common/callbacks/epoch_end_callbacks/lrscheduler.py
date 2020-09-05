class LRScheduler():

    def __init__(self, config):

        # take lr steps and decays from config
        self.steps = config.TRAIN.LR_STEPS
        self.decay = config.TRAIN.LR_DECAY

    def __call__(self, epoch=0, optimizer=None, **kwargs):

        if epoch in self.steps:
            # decay the LR for all param groups in optimizer
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * self.decay

class LRSchedulerPolicy():

    def __init__(self, config):

        # take lr steps and decays from config
        self.steps = config.POLICY.LR_STEPS
        self.decay = config.POLICY.LR_DECAY

    def __call__(self, epoch=0, policy_optimizer=None, **kwargs):

        if epoch in self.steps:
            # decay the LR for all param groups in optimizer
            for param_group in policy_optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * self.decay

