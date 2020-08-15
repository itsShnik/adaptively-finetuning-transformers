import matplotlib.pyplot as plt
import seaborn as sns
import torch
import wandb

class VisualizationPlotter:

    def __init__(self):
        pass

    def __call__(self, finetune_strategy, policy, policy_max, epoch, mode='train'):
        
        if finetune_strategy in ['SpotTune_Block', 'BlockDrop', 'BlockReplace']:

            assert int(policy.size(0)) == 12, f"{finetune_strategy} policy doesn't match required dimension 12!"

            # we need to create a lineplot
            # scale policy
            policy = policy / policy_max

            # create a matplotlib figure
            plt.style.use('seaborn-whitegrid')
            fig = plt.figure()
            ax = plt.axes()

            x = ['BL_' + str(k) for k in range(1,13)]

            plt.plot(x, list(policy))
            ax.set_xticklabels(x)
            plt.xlabel('Layers')
            plt.ylabel('Finetuned/Used Fraction')
            plt.ylim(0,1)
            plt.title(f'{finetune_strategy}_{mode}_Epoch_{epoch}')

            # just pass this plt to wandb.log while integrating with wandb
            plt.savefig('visualizations/{}_{}_epoch_{}.png'.format(finetune_strategy, mode, epoch))
            wandb.log({"{} Finetuning Fraction: {}".format(finetune_strategy, mode):plt})
            plt.close()

        elif finetune_strategy == 'SpotTune':

            assert int(policy.size(0)) == 180, "SpotTune policy doesn't match required dimension 180"

            policy = policy / policy_max

            policy =  policy.view(12, 15)

            xlabels = ['AH_' + str(k) for k in range(1,13)]
            xlabels.append('AO')
            xlabels.append('INT')
            xlabels.append('OP')

            ylabels = ['BL_' + str(k) for k in range(1,13)]

            # use seaborn plots
            ax = sns.heatmap(policy, xticklabels=xlabels, yticklabels=ylabels, cmap=sns.cm.rocket_r)
            plt.sca(ax)
            plt.savefig('visualizations/spottune_epoch_{}_{}.png'.format(mode,epoch))
            wandb.log({'SpotTune Finetuning Fraction: {}'.format(mode):wandb.Image(plt)})
            plt.close()
