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

def deterministic_policy_loss(policy, scale):
    """
    Policy: Action Vector
    Scale: The multiplying factor for loss
    """

    # first obtain the fraction of data using each block
    fraction = policy.mean(0)

    # to avoid log zero add a very small value to fraction
    fraction = fraction + 1e-9

    # calculate the loss
    loss = (-1 * fraction * torch.log(fraction)).sum(0)

    # scale the loss
    loss *= scale

    return loss

def log_fraction_loss(policy, scale):
    """
    Policy: Action Vector
    Scale: The multiplying factor for loss
    """

    # Find the fraction of blocks used per example
    fraction = policy.mean(-1)

    # Find the squared fraction
    sq_fraction = fraction ** 2

    # Subtract from 1 because we have to minimize the sq_fraction
    sq_fraction = 1 - sq_fraction

    # Take log and sum
    loss = -(torch.log(sq_fraction).sum(0))

    # scale the loss
    loss *= scale

    return loss
