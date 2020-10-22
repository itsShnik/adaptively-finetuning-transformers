# Adaptively Finetuning VLBERT

![VL-BERT Architecture](figs/pretrain.png)

## Data, Environment Setup and Running Instructions

Detailed instructions for data and environment setup, installing the pre-trained models and executing the code can be found in [vlbert\_readme.md](./vlbert_readme.md). Currently, this repository supports the adaptive finetuning experiments on the VQAv2 and VQA-CPv2 datasets only.

Summary: The following command can be used to execute the code on VQAv2 dataset with distributed training on multiple GPUs.

```sh
./scripts/dist_run_single.sh <num_gpus> vqa/train_end2end.py cfgs/vqa/base_4X16G_fp32.yaml output
```
The checkpoints will be stored in the ``output/`` directory.

## Results and Visualizations

### SpotTune and SpotTune\_Block

| Finetune Strategy  | Train Accuracy | Val Accuracy | Generalization Error | Params                |
| ------------------ | -------------- | ------------ | -------------------- | --------------------- |
| Pretrained Frozen  | 61.47          | 59.57        | 1.90					| 30 (T) + 85 (NT)      |
| Last 4 Trainable   | 76.92          | 67.68        | 9.24 			    | 58 (T) + 57 (NT)      |
| Frozen (3,4,10,11) | 80.78          | 67.79        | 11.99			    | 87 (T) + 28 (NT)      |
| Standard           | 83.40          | 68.22        | 15.18			    | 115 (T)               |
| SpotTune\_Block    | 80.54          | 67.90        | 12.64			    | 115 + 34 (T), 85 (NT) |
| SpotTune           | 81.21          | 68.14        | 13.07			    | 115 + 34 (T), 85 (NT) |

### Visualizations for SpotTune

Sample visualization for SpotTune strategy on VLBERT-Base trained on VQAv2 dataset.

| Epoch Number | Usage of Finetuned Blocks |
| ------------ | ------------------------- |
| 5 | ![Usage of finetuned blocks](visualizations/SpotTune_5.png?raw=true) |

The darker the tone of the above heatmaps is, the more finetuning is done for each of the component.

### Visualizations for SpotTune\_Block

Sample visualization for SpotTune\_Block strategy on VLBERT-Base trained on VQAv2 dataset.

| Epoch Number | Usage of Finetuned Blocks |
| ------------ | ------------------------- |
| 5 | ![Usage of finetuned blocks](visualizations/spottune_block_epoch_0.png?raw=true) |

### BlockDrop Results

| Experiments                                                                  | Training Acc | Val Acc |
| ---------------------------------------------------------------------------- | ------------ | ------- |
| VLBERT-Base [12 Layer][Pretrained] |              | 68.22   |
| VLBERT [4 Layer][Initialized from pretrained VLBERT-Base] | 77.09        | 66.23   |
| Blockdrop on VLBERT-Base [No additional loss][All 12 blocks used] | 81.69        | 68.11   |
| Blockdrop on VLBERT-Base [Constrain K loss][4 blocks used] | 73.99        | 66.26   |
| Blockdrop on VLBERT-Base [Constrain K loss][Global decision at inference] | 77.25        | 66.80   |

### Visualizations for BlockDrop

Sample visualization for BlockDrop strategy on VLBERT-Base trained on VQAv2 dataset.

| Epoch Number | Fraction of Data that uses the Block|
| ------------ | ------------------------- |
| 5 | ![Usage of finetuned blocks](visualizations/BlockDrop_val_epoch_4.png?raw=true) |

## Wandb

I have experimented with a lot of other adaptive finetuning methods, on other datasets and having different visualizations. I used wandb to log all my visualizations and training data. The link for the project can be found [here](https://wandb.ai/shnik/adaptive-finetuning?workspace=user-shnik).
