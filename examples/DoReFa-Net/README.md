Official code and model for the paper:

+ [DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](http://arxiv.org/abs/1606.06160).

It also contains an implementation of the following papers:
+ [Binary Weight Network](https://arxiv.org/abs/1511.00363), with (W,A,G)=(1,32,32).
+ [Trained Ternary Quantization](https://arxiv.org/abs/1612.01064), with (W,A,G)=(t,32,32).
+ [Binarized Neural Networks](https://arxiv.org/abs/1602.02830), with (W,A,G)=(1,1,32).

Alternative link to this page: [http://dorefa.net](http://dorefa.net)

## Results:
This is a good set of baselines for research in model quantization.
These quantization techniques, when applied on AlexNet, achieves the following ImageNet performance in this implementation:

| Model   | Bit Width <br/> (weights, activations, gradients) | Top 1 Validation Error <sup>[1](#ft1)</sup>   |Lego-Net|

|:-----------------:|:--------------------:|:----------:|:-----:|

| Full Precision<sup>[2](#ft2)</sup> | 32,32,32                                          | 40.3%    |
| TTQ                                | t,32,32                                           | 42.0%    |
| BWN                                | 1,32,32                                           | 44.3%  | 42.82%
| BNN                                | 1,1,32                                            | 51.5%  |
| DoReFa                             | 8,8,8                                             | 42.0%   |
| DoReFa                             | 1,2,32                                            | 46.6%    |
| DoReFa                             | 1,2,6                                             | 46.8%   |
| DoReFa                             | 1,2,4                                             | 54.0%    |

 <a id="ft1">1</a>: These numbers were obtained by training on 8 GPUs with a total batch size of 256 (otherwise the performance may become slightly different).
The DoReFa-Net models reach slightly better performance than our paper, due to
more sophisticated augmentations.

 <a id="ft2">2</a>: Not directly comparable with the original AlexNet. Check out
 [../ImageNetModels](../ImageNetModels) for a more faithful implementation of the original AlexNet.

