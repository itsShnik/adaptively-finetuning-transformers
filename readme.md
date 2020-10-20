# Adaptive Finetuning of Transformers

This repository explores adaptively finetuning large pre-trained transformers. The experiments are conducted on vision and language models -- VLBERT and LXMERT which are based on single-stream and two stream architechtures respectively.

## What are Transformers?

Transformers are deep neural networks built upon stacked multi-headed attention mechanisms. Transformers were first introduced in \[[1][1]\] for the tasks of machine translation. Since then, transformers have been widely used in pre-training of generic representations in NLP \[[2][2], [3][3]\], vision and language \[[4][4], [5][5], [6][6]\] and very recently in computer vision \[[7][7]\] as well. The rise of using transformer is attributed to the immense success these attention based networks have recieved for tasks in almost every modality. Another reason is the flexible architecture that can be used for almost any kind of input structures.

Architecture of a transformer encoder is depicted in the figure below.

![Transformer Architecture](images/transformer.png?raw=true)

## What is Finetuning?

Finetuning is a widely used method for transfer learning which is a paradigm to transfer the knowledge gained by machine learning models from a task/dataset to another (usually smaller).  

When finetuning a pre-trained model on a smaller dataset, the model is initialized by the pre-trained weights and the weights are updated by optimizing for accuracy on the smaller dataset.

## What do you mean "Adaptive"?

The experiments presented in this repository choose the parts of the pre-trained model to finetune/drop based on each instance (input). It is "adaptive" in the sense that the architecture is different for each of the input samples. The decision to choose the parts is made on the basis of a policy network which is very small when compared to the original model.

Adaptive finetuning has been previously explored for residual networks \[[8][8], [9][9]\]. The policy network can be optimized in specific ways to improve the efficiency, accuracy, generalization of the models.

## How to use this Repository?

The experiments presented are conducted on VLBERT and LXMERT. Detailed instructions to reproduce the experiments, comparisons and results are shown in the respective folders ```VLBERT``` and ```LXMERT```. Additionally, I have provided the links for [Wandb][10] workspaces for experiments on both the architectures. You can find the results, visualizations, training procedures, configs etc. in detail there.

## References

1. [Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. In Advances in neural information processing systems (pp. 5998-6008).][1]
2. [Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.][2]
3. [Yang, Z., Dai, Z., Yang, Y., Carbonell, J., Salakhutdinov, R. R., & Le, Q. V. (2019). Xlnet: Generalized autoregressive pretraining for language understanding. In Advances in neural information processing systems (pp. 5753-5763).][3]
4. [Lu, J., Batra, D., Parikh, D., & Lee, S. (2019). Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks. In Advances in Neural Information Processing Systems (pp. 13-23).][4]
5. [Su, W., Zhu, X., Cao, Y., Li, B., Lu, L., Wei, F., & Dai, J. (2019). Vl-bert: Pre-training of generic visual-linguistic representations. arXiv preprint arXiv:1908.08530.][5]
6. [Tan, H., & Bansal, M. (2019). Lxmert: Learning cross-modality encoder representations from transformers. arXiv preprint arXiv:1908.07490.][6]
7. [[Under Review] An image is worth 16x16 words: Transformers for Image Recognition at Scale][7]
8. [Guo, Y., Shi, H., Kumar, A., Grauman, K., Rosing, T., & Feris, R. (2019). Spottune: transfer learning through adaptive fine-tuning. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 4805-4814).][8]
9. [Wu, Z., Nagarajan, T., Kumar, A., Rennie, S., Davis, L. S., Grauman, K., & Feris, R. (2018). Blockdrop: Dynamic inference paths in residual networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 8817-8826).][9]

---

[1]: https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf
[2]: https://arxiv.org/pdf/1810.04805.pdf
[3]: https://papers.nips.cc/paper/8812-xlnet-generalized-autoregressive-pretraining-for-language-understanding.pdf
[4]: http://papers.nips.cc/paper/8297-vilbert-pretraining-task-agnostic-visiolinguistic-representations-for-vision-and-language-tasks.pdf
[5]: http://papers.nips.cc/paper/8297-vilbert-pretraining-task-agnostic-visiolinguistic-representations-for-vision-and-language-tasks.pdf
[6]: https://arxiv.org/pdf/1908.07490
[7]: https://openreview.net/pdf?id=YicbFdNTTy
[8]: http://openaccess.thecvf.com/content_CVPR_2019/papers/Guo_SpotTune_Transfer_Learning_Through_Adaptive_Fine-Tuning_CVPR_2019_paper.pdf
[9]: http://openaccess.thecvf.com/content_cvpr_2018/papers/Wu_BlockDrop_Dynamic_Inference_CVPR_2018_paper.pdf
