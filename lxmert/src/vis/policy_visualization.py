import matplotlib.pyplot as plt
import seaborn as sns
import torch
import wandb

class Visualization:
    def __init__(self, finetune_strategy):
        self.finetune_strategy = finetune_strategy

    def plot(self, policy, policy_max, epoch=0, mode='train'):

        # first scale to 0 and 1
        policy = policy / policy_max

        # create a matplotlib figure
        plt.style.use('seaborn-whitegrid')
        fig = plt.figure()
        ax = plt.axes()

        # create axis values
        x = [*range(0, policy.size(0))]

        plt.plot(x, list(policy))
        plt.xlabel('Block Numbers')
        plt.ylabel('Finetuned Fraction')
        plt.title(f'{self.finetune_strategy}_Epoch_{epoch}')
        plt.ylim(0,1.05)

        # just pass this plt to wandb.log while integrating with wandb
        # currently saving locally only
        plt.savefig('visualizations/{}_{}_epoch_{}.png'.format(self.finetune_strategy, mode, epoch))
        plt.close()
