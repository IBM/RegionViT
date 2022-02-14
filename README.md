# RegionViT: Regional-to-Local Attention for Vision Transformers

This repository is the official implementation of RegionViT: Regional-to-Local Attention for Vision Transformers. [ArXiv](https://arxiv.org/abs/2106.02689) 

We provided the codes for [Image Classification](#image-classification) and [Object Detection](#object-detection).

If you use the codes and models from this repo, please cite our work. Thanks!

```
@inproceedings{
    chen2021regionvit,
    title={{RegionViT: Regional-to-Local Attention for Vision Transformers}},
    author={Chun-Fu (Richard) Chen and Rameswar Panda and Quanfu Fan},
    booktitle={ArXiv},
    year={2021}
}
```

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

We provide models trained on ImageNet1K. Models can be found [here](https://github.com/IBM/RegionViT/releases/tag/weights-v0.1).

| Name | Acc@1 | #FLOPs | #Params | URL |
| --- | --- | --- | --- | --- |
| RegionViT-Ti | 80.4 | 2.4 | 13.8M | [model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/RegionViT-Ti.pth) |
| RegionViT-S | 82.6 | 5.3 | 30.6M| [model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/RegionViT-S.pth) |
| RegionViT-M | 83.1 | 7.4 | 41.2M | [model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/RegionViT-M.pth) |
| RegionViT-B | 83.2 | 13.0 | 72.7M | [model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/RegionViT-B.pth) |


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

Then, to train RetinaNet RegionViT-S on MS COCO with 1x schedule:

```shell script

python main_detection.py --num-gpus 8 --resume --config-file detection/configs/retinanet_regionvit_FPN_1x.yaml MODEL.BACKBONE.REGIONVIT regionvit_small_224 MODEL.WEIGHTS /path/to/pretrained_model OUTPUT_DIR /path/to/log_folder
```

Model names of other models are `regionvit_base_224`, `regionvit_small_w14_224`, etc. Supported models can be found [here](./regionvit/regionvit.py)

### Model Zoo

We provide models trained on MS COCO with MaskRCNN and RetinaNet. Models can be found [here](https://github.com/IBM/RegionViT/releases/tag/weights-v0.1).

#### MaskRCNN

| Name | #Params (M) | #FLOPs (G) | box mAP (1x) | mask mAP (1x) | box mAP (3x) | mask mAP (3x) | url |
| --- | --- | --- | --- | --- | ---| --- | --- |
| RegionViT-S  | 50.1 | 171.3 | 42.5 | 39.5 | 46.3 | 42.3 | [1x model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/MaskRCNN_1x_RegionViT-S.pth) <br /> [3x model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/MaskRCNN_3x_RegionViT-S.pth) |
| RegionViT-S+ | 50.9 | 182.9 | 43.5 | 40.4 | 47.3 | 43.4 | [1x model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/MaskRCNN_1x_RegionViT-S+.pth) <br /> [3x model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/MaskRCNN_3x_RegionViT-S+.pth) |
| RegionViT-S+ (w/ PEG) | 50.9 | 183.0 | 44.2 | 40.8 | 47.6 | 43.4 | [1x model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/MaskRCNN_1x_RegionViT-S+peg.pth) <br /> [3x model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/MaskRCNN_3x_RegionViT-S+peg.pth) |
| RegionViT-B  | 92.2 | 287.9 | 43.5 | 40.1 | 47.2 | 43.0 | [1x model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/MaskRCNN_1x_RegionViT-B.pth) <br /> [3x model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/MaskRCNN_3x_RegionViT-B.pth) |
| RegionViT-B+ | 93.2 | 307.1 | 44.5 | 41.0 | 48.1 | 43.5 | [1x model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/MaskRCNN_1x_RegionViT-B+.pth) <br /> [3x model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/MaskRCNN_3x_RegionViT-B+.pth) |
| RegionViT-B+ (w/ PEG) | 93.2 | 307.2 | 45.4 | 41.6 | 48.3 | 43.5 | [1x model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/MaskRCNN_1x_RegionViT-B+peg.pth) <br /> [3x model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/MaskRCNN_3x_RegionViT-B+peg.pth) |
| RegionViT-B+ (w/ PEG) dagger | 93.2 | 464.4 | 46.3 | 42.4 | 49.2 | 44.5 | [1x model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/MaskRCNN_1x_RegionViT-B+peg_dagger.pth) <br /> [3x model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/MaskRCNN_3x_RegionViT-B+peg_dagger.pth) |


#### RetinaNet

| Name | #Params (M) | #FLOPs (G) | box mAP (1x) | box mAP (3x) | url |
| --- | --- | --- | --- | --- | ---| 
| RegionViT-S  | 40.8 | 192.6 | 42.2 | 45.8  | [1x model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/RetinaNet_1x_RegionViT-S.pth) <br /> [3x model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/RetinaNet_3x_RegionViT-S.pth) |
| RegionViT-S+ | 41.5 | 204.2 | 43.1 | 46.9 | [1x model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/RetinaNet_1x_RegionViT-S+.pth) <br /> [3x model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/RetinaNet_3x_RegionViT-S+.pth) |
| RegionViT-S+ (w/ PEG) | 41.6 | 204.3 | 43.9 | 46.7  | [1x model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/RetinaNet_1x_RegionViT-S+peg.pth) <br /> [3x model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/RetinaNet_3x_RegionViT-S+peg.pth) |
| RegionViT-B  | 83.4 | 308.9 | 43.3 | 46.1 | [1x model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/RetinaNet_1x_RegionViT-B.pth) <br /> [3x model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/RetinaNet_3x_RegionViT-B.pth) |
| RegionViT-B+ | 84.4 | 328.1 | 44.2 | 46.9| [1x model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/RetinaNet_1x_RegionViT-B+.pth) <br /> [3x model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/RetinaNet_3x_RegionViT-B+.pth) |
| RegionViT-B+ (w/ PEG) | 84.5 | 328.2 | 44.6 | 46.9  | [1x model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/RetinaNet_1x_RegionViT-B+peg.pth) <br /> [3x model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/RetinaNet_3x_RegionViT-B+peg.pth) |
| RegionViT-B+ (w/ PEG) dagger | 84.5 | 506.4 | 46.1 | 48.2  | [1x model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/RetinaNet_1x_RegionViT-B+peg_dagger.pth) <br /> [3x model](https://github.com/IBM/RegionViT/releases/download/weights-v0.1/RetinaNet_3x_RegionViT-B+peg_dagger.pth) |

