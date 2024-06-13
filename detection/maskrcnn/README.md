# Mask R-CNN with TransNeXt backbone on COCO

## Model Zoo

**COCO object detection and instance segmentation results using the Mask R-CNN method:**

| Backbone | Pretrained Model| Lr Schd| box mAP | mask mAP | #Params | Download |Config| Log |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|:---:|:---:|
| TransNeXt-Tiny | [ImageNet-1K](https://huggingface.co/DaiShiResearch/transnext-tiny-224-1k/resolve/main/transnext_tiny_224_1k.pth?download=true) |1x|49.9|44.6|47.9M|[model](https://huggingface.co/DaiShiResearch/maskrcnn-transnext-tiny-coco/resolve/main/mask_rcnn_transnext_tiny_fpn_1x_coco_in1k.pth?download=true)|[config](/detection/maskrcnn/configs/mask_rcnn_transnext_tiny_fpn_1x_coco.py)|[log](https://huggingface.co/DaiShiResearch/maskrcnn-transnext-tiny-coco/raw/main/mask_rcnn_transnext_tiny_fpn_1x_coco_in1k.log.json)|
| TransNeXt-Small | [ImageNet-1K](https://huggingface.co/DaiShiResearch/transnext-small-224-1k/resolve/main/transnext_small_224_1k.pth?download=true) |1x|51.1|45.5|69.3M|[model](https://huggingface.co/DaiShiResearch/maskrcnn-transnext-small-coco/resolve/main/mask_rcnn_transnext_small_fpn_1x_coco_in1k.pth?download=true)|[config](/detection/maskrcnn/configs/mask_rcnn_transnext_small_fpn_1x_coco.py)|[log](https://huggingface.co/DaiShiResearch/maskrcnn-transnext-small-coco/raw/main/mask_rcnn_transnext_small_fpn_1x_coco_in1k.log.json)|
| TransNeXt-Base | [ImageNet-1K](https://huggingface.co/DaiShiResearch/transnext-base-224-1k/resolve/main/transnext_base_224_1k.pth?download=true) |1x|51.7|45.9|109.2M|[model](https://huggingface.co/DaiShiResearch/maskrcnn-transnext-base-coco/resolve/main/mask_rcnn_transnext_base_fpn_1x_coco_in1k.pth?download=true)|[config](/detection/maskrcnn/configs/mask_rcnn_transnext_base_fpn_1x_coco.py)|[log](https://huggingface.co/DaiShiResearch/maskrcnn-transnext-base-coco/raw/main/mask_rcnn_transnext_base_fpn_1x_coco_in1k.log.json)|
* *When we checked the training logs, we found that the mask mAP and other detailed performance of the Mask R-CNN using the TransNeXt-Tiny backbone were **even better** than reported in the paper (versions V1 and V2). We have already fixed this in version V3  (it should be a data entry error).*

## Requirements

    pip install -r requirements.txt

## Data preparation

    cd /path/to/current_folder
    ln -s /your/path/to/coco-dataset ./data

## Evaluation
To evaluate Mask R-CNN models with TransNeXt backbone on COCO val, you can use the following command:

    bash dist_test.sh <config-file> <checkpoint-path> <gpu-num> --eval bbox segm

For example, to evaluate the TransNeXt-Tiny on a single GPU:
    
    bash dist_test.sh ./configs/mask_rcnn_transnext_tiny_fpn_1x_coco.py /path/to/checkpoint_file 1 --eval bbox segm
    
For example, to evaluate the TransNeXt-Tiny on 8 GPUs:
    
    bash dist_test.sh ./configs/mask_rcnn_transnext_tiny_fpn_1x_coco.py /path/to/checkpoint_file 8 --eval bbox segm


## Training
In order to train Mask R-CNN models with TransNeXt backbone on the COCO dataset, first, you need to fill in the path of your downloaded pretrained checkpoint in `./configs/<config-file>`. Specifically, change it to:
    
    pretrained=<path-to-checkpoint>, 

After setting up, to train TransNeXt on COCO dataset, you can use the following command:
    
    bash dist_train.sh <config-file> <gpu-num> 

For example, to train the TransNeXt-Tiny on 8 GPUs, with a total batch-size of 16:

    bash dist_train.sh ./configs/mask_rcnn_transnext_tiny_fpn_1x_coco.py 8

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