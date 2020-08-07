import torch
import torch.nn.functional as F

def constrain_k_loss(policy, num_blocks, scale):
    """
    policy is the action vector
    num_blocks is the number of global blocks
    scale is the multiplying factor for loss
    """

    # first obtain the sum of fraction
    sum_of_fraction = policy.mean(0).sum(0)

    # now calculate the l2 loss
    loss = (sum_of_fraction - num_blocks) ** 2

    # scale the loss
    loss *= scale

    return loss
