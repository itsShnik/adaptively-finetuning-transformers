# Adaptively Finetuning VLBERT

![VL-BERT Architecture](figs/pretrain.png)

## Data, Environment Setup and Running Instructions

Detailed instructions for data and environment setup, installing the pre-trained models and executing the code can be found in [vlbert\_readme.md][1]. Currently, this repository supports the adaptive finetuning experiments on the VQAv2 and VQA-CPv2 datasets only.

Summary: The following command can be used to execute the code on VQAv2 dataset with distributed training on multiple GPUs.

```sh
./scripts/dist_run_single.sh <num_gpus> vqa/train_end2end.py cfgs/vqa/base_4X16G_fp32.yaml output
```
The checkpoints will be stored in the ``output/`` directory.

## Explanation of the Finetuning Setting

VLBERT is a model that allows us to pre-train generic vision and language representations using semi-supervised learning on large corpus. Once the model is pre-trained, it can then be finetuned on any downstream vision and language dataset (usually by replacing the classifier head) to achieve robust performance. In particular, VLBERT is pre-trained on a combination of Conceptual Captions and BookCorpus datasets.

When the standard method for finetuning is used, the whole network is finetuned in an ordinary fashion. Using the standard finetuning method, one can reproduce the results of VLBERT on downstream tasks as provided in the paper.

## Adaptive Finetuning Basics

We explore several different adaptive finetuning strategies in this repository. One thing that is common to all the strategies is the use of a policy network to determine which parts of the model to finetune/drop based on the input images-text pair. The chosen policy network is relatively very small when compared to the original VLBERT network. The policy network is optimized using Gumbel Softmax which relieves the argmax constraints to softmax while backpropagation.

## Adaptive Finetuning Strategies

1. __SpotTune\_Block__: The transformer encoder architecture of VLBERT is made of stacked self-attention blocks. When using this strategy, the policy network decides for each of the blocks whether to use the pre-trained version or the (continuously) finetuned version. For instance, there are 12 attention blocks used in the VLBERT-Base architecture, therefore, the policy network will make 12 binary decisions, one for each block. The policy is depicted in the diagram below.

2. __SpotTune__: Each block of the transformer encoder architecture consists of several components -- _n_ attention heads, 1 attention output FF layer, 1 intermediate FF layer and 1 output FF layer. For an attention method with 12 heads, there are total 15 decisions to be made per block in case of SpotTune strategy. The policy is depicted in the diagram below.

3. __BlockDrop__: This strategy is slightly different from the other two. In this case, we drop some parts of the network adaptively, therefore, having a more efficient architecture at the inference time.

## Results and Visualizations
