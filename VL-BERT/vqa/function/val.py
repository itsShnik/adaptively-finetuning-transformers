from collections import namedtuple
import torch
from common.trainer import to_cuda
from common.gumbel_softmax import gumbel_softmax
from common.callbacks.epoch_end_callbacks.visualization_plotter import VisualizationPlotter

# policy vector shapes
PolicyVec = {'SpotTune':180,
        'SpotTune_Block':12,
        'BlockDrop':12}

# create a visualization object
vis = VisualizationPlotter()

@torch.no_grad()
def do_validation(net, val_loader, metrics, label_index_in_batch, epoch_num=0, finetune_strategy='standard', policy_net=None, policy_optimizer=None, global_decision=False, policy_decisions=None, policy_total=None):
    net.eval()
    if finetune_strategy in PolicyVec:
        policy_net.eval()
        policy_save = torch.zeros(PolicyVec[finetune_strategy]).cpu()
        policy_max = 0

        # check if we have to make a global decision
        if global_decision:
            # calculate the policy
            policy_decisions = policy_decisions / policy_total
            policy_init = (policy_decisions > 0.5).float()


    metrics.reset()
    for nbatch, batch in enumerate(val_loader):
        batch = to_cuda(batch)
        label = batch[label_index_in_batch]
        datas = [batch[i] for i in range(len(batch)) if i != label_index_in_batch % len(batch)]

        if finetune_strategy in PolicyVec:
            if not global_decision:
                policy_vector = policy_net(*datas)
                policy_action = gumbel_softmax(policy_vector.view(policy_vector.size(0), -1, 2))
                policy = policy_action[:,:,1]
            else:
                # repeat to match the batch size
                policy = policy_init.repeat(batch[1].size(0), 1)
            policy_save = policy_save + policy.clone().detach().cpu().sum(0)
            policy_max += policy.size(0)
            outputs = net(*datas, policy)
        else:
            outputs = net(*datas)
        outputs.update({'label': label})
        metrics.update(outputs)

    # plot val visualizations
    print("Plotting val visualizations")
    vis(finetune_strategy, policy_save, policy_max, epoch_num, mode='val')

