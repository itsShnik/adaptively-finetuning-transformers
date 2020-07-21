from collections import namedtuple
import torch
from common.trainer import to_cuda


@torch.no_grad()
def do_validation(net, val_loader, metrics, label_index_in_batch, policy_net=None, policy_optimizer=None):
    net.eval()
    if policy_net is not None:
        policy_net.eval()

    metrics.reset()
    for nbatch, batch in enumerate(val_loader):
        batch = to_cuda(batch)
        label = batch[label_index_in_batch]
        datas = [batch[i] for i in range(len(batch)) if i != label_index_in_batch % len(batch)]

        if policy_net is not None:
            policy_vector = policy_net(*datas)
            policy_action = gumbel_softmax(policy_vector.view(policy_vector.size(0), -1, 2))
            policy = policy_action[:,:,1]
            outputs = net(*datas, policy)
        else:
            outputs = net(*datas)
        outputs.update({'label': label})
        metrics.update(outputs)

