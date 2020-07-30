import matplotlib.pyplot as plt
import seaborn as sns
import torch
import wandb

class VisualizationPlotter:

    def __init__(self):
        pass

    def __call__(self, finetune_strategy, policy, policy_max, epoch):
        
        if finetune_strategy == 'SpotTune_Block':

            assert int(policy.size(0)) == 12, "SpotTune_Block policy doesn't match required dimension 12!"

            # we need to create a lineplot
            # scale policy
            policy = policy / policy_max

            # create a matplotlib figure
            plt.style.use('seaborn-whitegrid')
            fig = plt.figure()
            ax = plt.axes()

            x = ['BL_' + str(k) for k in range(1,13)]

            plt.plot(x, list(policy))
            plt.xlabel('Layers')
            plt.ylabel('Finetuned Fraction')
            plt.title(f'SpotTune_Block_Epoch_{epoch}')

            # just pass this plt to wandb.log while integrating with wandb
            plt.savefig('visualizations/spottune_block_epoch_{}.png'.format(epoch))
            wandb.log({"SpotTune_Block Finetuning Fraction Epoch {}".format(epoch):plt})
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
            plt.savefig('visualizations/spottune_epoch_{}.png'.format(epoch))
            wandb.log({'SpotTune Finetuning Fraction Epoch {}'.format(epoch):wandb.Image(plt)})
            plt.close()
