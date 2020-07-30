import torch
import torch.nn as nn
import torch.nn.functional as F
import time

def sample_gumbel(shape, eps=1e-20):
    U = torch.cuda.FloatTensor(shape).uniform_()
    return -(torch.log(-torch.log(U + eps) + eps))

def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature = 5):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y

if __name__=='__main__':
    logits = torch.randn(1, 12, 2)
    print(logits)

    # make requires_grad true
    logits.requires_grad = True

    logits = logits.to('cuda:0')
    policy = gumbel_softmax(logits)
    print("The shape of policy is : ", policy.size())
    print(policy)
    print(policy.requires_grad)
