# COCO Object detection with TransNeXt

## Getting started

This repository is the official PyTorch implementation of TransNeXt for COCO object detection.

Our code is built on [MMDetection](https://github.com/open-mmlab/mmdetection). The Mask R-CNN method is built on
MMDetection version 2.28.2, while the DINO method is built on MMDetection version 3.0.0.

Since MMDetection is no longer compatible with the previous version of the configuration file format after 3.0.0,
different environments need to be built for the two methods. The `requirements.txt` can be found in their respective
folders.

## Model Zoo

**COCO object detection and instance segmentation results using the Mask R-CNN method:**

| Backbone | Pretrained Model| Lr Schd| box mAP | mask mAP | #Params | Download |Config| Log |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|:---:|:---:|
| TransNeXt-Tiny | [ImageNet-1K](https://huggingface.co/DaiShiResearch/transnext-tiny-224-1k/resolve/main/transnext_tiny_224_1k.pth?download=true) |1x|49.9|44.6|47.9M|[model](https://huggingface.co/DaiShiResearch/maskrcnn-transnext-tiny-coco/resolve/main/mask_rcnn_transnext_tiny_fpn_1x_coco_in1k.pth?download=true)|[config](/detection/maskrcnn/configs/mask_rcnn_transnext_tiny_fpn_1x_coco.py)|[log](https://huggingface.co/DaiShiResearch/maskrcnn-transnext-tiny-coco/raw/main/mask_rcnn_transnext_tiny_fpn_1x_coco_in1k.log.json)|
| TransNeXt-Small | [ImageNet-1K](https://huggingface.co/DaiShiResearch/transnext-small-224-1k/resolve/main/transnext_small_224_1k.pth?download=true) |1x|51.1|45.5|69.3M|[model](https://huggingface.co/DaiShiResearch/maskrcnn-transnext-small-coco/resolve/main/mask_rcnn_transnext_small_fpn_1x_coco_in1k.pth?download=true)|[config](/detection/maskrcnn/configs/mask_rcnn_transnext_small_fpn_1x_coco.py)|[log](https://huggingface.co/DaiShiResearch/maskrcnn-transnext-small-coco/raw/main/mask_rcnn_transnext_small_fpn_1x_coco_in1k.log.json)|
| TransNeXt-Base | [ImageNet-1K](https://huggingface.co/DaiShiResearch/transnext-base-224-1k/resolve/main/transnext_base_224_1k.pth?download=true) |1x|51.7|45.9|109.2M|[model](https://huggingface.co/DaiShiResearch/maskrcnn-transnext-base-coco/resolve/main/mask_rcnn_transnext_base_fpn_1x_coco_in1k.pth?download=true)|[config](/detection/maskrcnn/configs/mask_rcnn_transnext_base_fpn_1x_coco.py)|[log](https://huggingface.co/DaiShiResearch/maskrcnn-transnext-base-coco/raw/main/mask_rcnn_transnext_base_fpn_1x_coco_in1k.log.json)|
* *When we checked the training logs, we found that the mask mAP and other detailed performance of the Mask R-CNN using the TransNeXt-Tiny backbone were **even better** than reported in the paper (versions V1 and V2). We have already fixed this in version V3  (it should be a data entry error).*

**COCO object detection results using the DINO method:**

| Backbone | Pretrained Model| scales | epochs | box mAP | #Params | Download |Config| Log |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|:---:|:---:|
| TransNeXt-Tiny | [ImageNet-1K](https://huggingface.co/DaiShiResearch/transnext-tiny-224-1k/resolve/main/transnext_tiny_224_1k.pth?download=true)|4scale | 12|55.1|47.8M|[model](https://huggingface.co/DaiShiResearch/dino-4scale-transnext-tiny-coco/resolve/main/dino_4scale_transnext_tiny_12e_in1k.pth?download=true)|[config](/detection/dino/configs/dino-4scale_transnext_tiny-12e_coco.py)|[log](https://huggingface.co/DaiShiResearch/dino-4scale-transnext-tiny-coco/raw/main/dino_4scale_transnext_tiny_12e_in1k.json)|
| TransNeXt-Tiny | [ImageNet-1K](https://huggingface.co/DaiShiResearch/transnext-tiny-224-1k/resolve/main/transnext_tiny_224_1k.pth?download=true)|5scale | 12|55.7|48.1M|[model](https://huggingface.co/DaiShiResearch/dino-5scale-transnext-tiny-coco/resolve/main/dino_5scale_transnext_tiny_12e_in1k.pth?download=true)|[config](/detection/dino/configs/dino-5scale_transnext_tiny-12e_coco.py)|[log](https://huggingface.co/DaiShiResearch/dino-5scale-transnext-tiny-coco/raw/main/dino_5scale_transnext_tiny_12e_in1k.json)|
| TransNeXt-Small | [ImageNet-1K](https://huggingface.co/DaiShiResearch/transnext-small-224-1k/resolve/main/transnext_small_224_1k.pth?download=true)|5scale | 12|56.6|69.6M|[model](https://huggingface.co/DaiShiResearch/dino-5scale-transnext-small-coco/resolve/main/dino_5scale_transnext_small_12e_in1k.pth?download=true)|[config](/detection/dino/configs/dino-5scale_transnext_small-12e_coco.py)|[log](https://huggingface.co/DaiShiResearch/dino-5scale-transnext-small-coco/raw/main/dino_5scale_transnext_small_12e_in1k.json)|
| TransNeXt-Base | [ImageNet-1K](https://huggingface.co/DaiShiResearch/transnext-base-224-1k/resolve/main/transnext_base_224_1k.pth?download=true)|5scale | 12|57.1|110M|[model](https://huggingface.co/DaiShiResearch/dino-5scale-transnext-base-coco/resolve/main/dino_5scale_transnext_base_12e_in1k.pth?download=true)|[config](/detection/dino/configs/dino-5scale_transnext_base-12e_coco.py)|[log](https://huggingface.co/DaiShiResearch/dino-5scale-transnext-base-coco/raw/main/dino_5scale_transnext_base_12e_in1k.json)|

## How to use

***The code & tutorial for the Mask R-CNN method are >> [here](maskrcnn) <<***

***The code & tutorial for the DINO method are >> [here](dino) <<***

## Acknowledgement

The released script for Object Detection with TransNeXt is built based on the [MMDetection](https://github.com/open-mmlab/mmdetection) and [timm](https://github.com/huggingface/pytorch-image-models) library.

## License

This project is released under the Apache 2.0 license. Please see the [LICENSE](/LICENSE) file for more information.


## Citation

If you find our work helpful, please consider citing the following bibtex. We would greatly appreciate a star for this
project.

    @InProceedings{shi2023transnext,
      author    = {Dai Shi},
      title     = {TransNeXt: Robust Foveal Visual Perception for Vision Transformers},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      month     = {June},
      year      = {2024},
      pages     = {17773-17783}
    }

