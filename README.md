# Vision GNN

ML with Graphs Project

Paper : Vision GNN : An Image is Worth Graph of Nodes
By Kai Han, Yunhe Wang, Jianyuan Guo, Yehui Tang and Enhua Wu. NeurIPS 2022. [[arXiv link]](https://arxiv.org/abs/2206.00272)

## Requirements

- Python 3.7

* PyTorch 1.7.0
* timm 0.3.2
* torchprofile 0.0.4
* Numpy
* tqdm
* pandas
* matplotlib
* wandb (optional)
* apex (optional)

## Pretrained models

- ViG (ImageNet Models - from the original paper)

| Model  | Params (M) | FLOPs (B) | Top-1 | Github Release                                                                           |
| ------ | ---------- | --------- | ----- | ---------------------------------------------------------------------------------------- |
| ViG-Ti | 7.1        | 1.3       | 73.9  | [Github Release](https://github.com/huawei-noah/Efficient-AI-Backbones/releases/tag/vig) |
| ViG-S  | 22.7       | 4.5       | 80.4  | [Github Release](https://github.com/huawei-noah/Efficient-AI-Backbones/releases/tag/vig) |
| ViG-B  | 86.8       | 17.7      | 82.3  | [Github Release](https://github.com/huawei-noah/Efficient-AI-Backbones/releases/tag/vig) |

- Pyramid ViG (ImageNet Models - from the original paper)

| Model          | Params (M) | FLOPs (B) | Top-1 | BaiduDisk URL                                                                  | Github Release                                                                                   |
| -------------- | ---------- | --------- | ----- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------ |
| Pyramid ViG-Ti | 10.7       | 1.7       | 78.5  | [[BaiduDisk]](https://pan.baidu.com/s/1Vrr-oXQeUFujaHKMC5sXIQ), Password: chae | [Github Release](https://github.com/huawei-noah/Efficient-AI-Backbones/releases/tag/pyramid-vig) |
| Pyramid ViG-S  | 27.3       | 4.6       | 82.1  | [[BaiduDisk]](https://pan.baidu.com/s/10MWZznvPIvGAiBtnwj7TRg), Password: 81mg | [Github Release](https://github.com/huawei-noah/Efficient-AI-Backbones/releases/tag/pyramid-vig) |
| Pyramid ViG-M  | 51.7       | 8.9       | 83.1  | [[BaiduDisk]](https://pan.baidu.com/s/1N3nviACOrY0XBC0FKoDL6g), Password: prd3 | [Github Release](https://github.com/huawei-noah/Efficient-AI-Backbones/releases/tag/pyramid-vig) |
| Pyramid ViG-B  | 82.6       | 16.8      | 83.7  | [[BaiduDisk]](https://pan.baidu.com/s/1b5OvPZXwcSwur2nuDAzf5Q), Password: rgm4 | [Github Release](https://github.com/huawei-noah/Efficient-AI-Backbones/releases/tag/pyramid-vig) |

## Evaluation

Data preparation follows the [official pytorch example](https://github.com/pytorch/examples/tree/main/imagenet)

- Evaluate example:

```
python train.py /path/to/imagenet/ --model pvig_s_224_gelu -b 128 --pretrain_path /path/to/pretrained/model/ --evaluate
```

## Training

- Training ViG on 4 GPUs:

```
python -m torch.distributed.launch --nproc_per_node=8 train.py /path/to/imagenet/ --model vig_s_224_gelu --sched cosine --epochs 100 --opt adamw -j 8 --warmup-lr 1e-6 --cutmix 1.0 --model-ema --model-ema-decay 0.99996 --aa rand-m9-mstd0.5-inc1 --color-jitter 0.4 --warmup-epochs 20 --opt-eps 1e-8 --repeated-aug --remode pixel --reprob 0.25 --amp --lr 2e-3 --weight-decay .05 --drop 0 --drop-path .1 -b 256 --output /path/to/save/models/
```

- Training Pyramid ViG on 4 GPUs:

```
python -m torch.distributed.launch --nproc_per_node=8 train.py /path/to/imagenet/ --model pvig_s_224_gelu --sched cosine --epochs 100 --opt adamw -j 8 --warmup-lr 1e-6 --model-ema --model-ema-decay 0.99996 --aa rand-m9-mstd0.5-inc1 --color-jitter 0.4 --warmup-epochs 20 --opt-eps 1e-8 --repeated-aug --remode pixel --reprob 0.25 --amp --lr 2e-3 --weight-decay .05 --drop 0 --drop-path .1 -b 256 --output /path/to/save/models/
```

## Acknowledgements

Borrowed a big chunk of the code from [Vision-GNN](https://github.com/jichengyuan/Vision_GNN)
Inspiration from [deep_gcns_torch](https://github.com/lightaime/deep_gcns_torch) and [timm](https://github.com/rwightman/pytorch-image-models).

```

```
