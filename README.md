# RegionViT: Regional-to-Local Attention for Vision Transformers

This repository is the official implementation of RegionViT: Regional-to-Local Attention for Vision Transformers. [ArXiv](https://arxiv.org/abs/2106.02689) 

We provided the codes for [Image Classification](#image-classification) and [Object Detection](#object-detection).

If you use the codes and models from this repo, please cite our work. Thanks!

@inproceedings{
    chen2021regionvit,
    title={{RegionViT: Regional-to-Local Attention for Vision Transformers}},
    author={Chun-Fu (Richard) Chen and Rameswar Panda and Quanfu Fan},
    booktitle={ArXiv},
    year={2021}
}

## Image Classification

### Installation

To install requirements:

```setup
pip install -r requirements.txt
```

### Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class/2
      img4.jpeg
```

### Model Zoo

We provide models pretrained on ImageNet1K.

| name | acc@1 | acc@5 | #params | url |
| --- | --- | --- | --- | --- |
| RegionViT-tiny | 72.2 | 91.1 | 5M | [model](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth) |
| RegionViT-small | 79.9 | 95.0 | 22M| [model](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth) |
| RegionViT-medium | 81.8 | 95.6 | 86M | [model](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth) |
| RegionViT-base | 74.5 | 91.9 | 6M | [model](https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth) |


### Training

To train RegionViT-S on ImageNet on a single node with 8 gpus for 300 epochs run:

```shell script

python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model regionvit_small_224 --batch-size 256 --data-path /path/to/imagenet
```

Model names of other models are `regionvit_tiny_224`, `regionvit_medium_224` and `regionvit_base_224`.

### Multinode training

Distributed training is available via Slurm and `submitit`:

To train RegionViT-S model on ImageNet on 4 nodes with 8 gpus each for 300 epochs:

```
python run_with_submitit.py --model regionvit_small_224 --data-path /path/to/imagenet --batch-size 256 --warmup-epochs 50
```

Note that: some slurm configurations might need to be changed based on your cluster.


### Evaluation

To evaluate a pretrained model on RegionViT-S:

```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model regionvit_small_224 --batch-size 256 --data-path /path/to/imagenet --eval --initial_checkpoint /path/to/checkpoint
```


## Object Detection

We performed the object detection based on [Detectron2](https://github.com/facebookresearch/detectron2) with some modifications.

The modified version can be found at https://github.com/chunfuchen/detectron2. 
The major difference is the data augmentation pipepile.

### Installation

Follows the installation guide on [Install.md](https://github.com/chunfuchen/detectron2/blob/master/INSTALL.md).

### Data preparation

Follows Detectron2 to setup MS COCO dataset. [Link](https://detectron2.readthedocs.io/en/latest/tutorials/builtin_datasets.html)

### Training

Before training, you will need to convert the pretrained model into Detectron2 format. We provide the script `tools/convert_cls_model_to_d2.py` for the conversion.

```shell script
python3 tools/convert_cls_model_to_d2.py --model /path/to/pretrained/model --ows 7 --nws 7
```

Then, to train RegionViT-S on MS COCO with 1x schedule:

```shell script

python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py --model regionvit_small_224 --batch-size 256 --data-path /path/to/imagenet
```

Model names of other models are `regionvit_tiny_224`, `regionvit_medium_224` and `regionvit_base_224`.



