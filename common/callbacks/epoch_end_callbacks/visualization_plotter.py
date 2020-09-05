#----------------------------------------
#--------- Python lib imports -----------
#----------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

class VisualizationPlotter():

    def __init__(self):
        pass

    def __call__(self, policy_decisions=None, policy_max=None, epoch=0, training_strategy='SpotTune', mode='train', **kwargs):
        
        if training_strategy in ['SpotTune']:

            print(f"Plotting {mode} visualizations!!")

            assert int(policy_decisions.size(0)) == 12, f"{training_strategy} policy doesn't match required dimension 12!"

            # we need to create a lineplot
            # scale policy
            policy_decisions = policy_decisions / policy_max

            # create a matplotlib figure
            plt.style.use('seaborn-whitegrid')
            fig = plt.figure()
            ax = plt.axes()

            x = ['Block_' + str(k) for k in range(1,13)]

            plt.plot(x, list(policy_decisions))
            ax.set_xticklabels(x)
            plt.xlabel('Layers')
            plt.ylabel('Finetuned/Used Fraction')
            plt.ylim(0,1)
            plt.title(f'{training_strategy}_{mode}_Epoch_{epoch}')

            # just pass this plt to wandb.log while integrating with wandb
            plt.savefig('visualizations/{}_{}_epoch_{}.png'.format(training_strategy, mode, epoch))
            wandb.log({"{} Finetuning Fraction: {}".format(training_strategy, mode):plt})
            plt.close()
