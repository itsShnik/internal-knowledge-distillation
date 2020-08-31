#----------------------------------------
#--------- OS related imports -----------
#----------------------------------------
import os
from pathlib import Path

#----------------------------------------
#--------- Torch related imports --------
#----------------------------------------
import torch

class Checkpoint():
    def __init__(self, config, val_metrics):
        self.config = config
        self.val_metrics = val_metrics
        self.save_path = self.set_up_logging_dir()

    def set_up_logging_dir(self):
        ckpt_dir_path = os.path.join('ckpts', self.config.VERSION)
        Path(ckpt_dir_path).mkdir(parents=True, exist_ok=True)
        return ckpt_dir_path

    def __call__(self, epoch, net, optimizer):
        # save the current epoch metrics
        curr_save_info = {
                'net_state_dict':net.state_dict(),
                'optim_state_dict':optimizer.state_dict()
                }

        torch.save(curr_save_info, os.path.join(self.save_path, f'epoch_{epoch}.pth'))

        if self.val_metrics.updated_best_val:
            print("Saving new best model...")
            torch.save(curr_save_info, os.path.join(self.save_path, f'best.pth'))
            print("Done!!")
