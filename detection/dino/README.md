# DINO with TransNeXt backbone on COCO

## Model Zoo

**COCO object detection results using the DINO method:**

| Backbone | Pretrained Model| scales | epochs | box mAP | #Params | Download |Config| Log |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|:---:|:---:|
| TransNeXt-Tiny | [ImageNet-1K](https://huggingface.co/DaiShiResearch/transnext-tiny-224-1k/resolve/main/transnext_tiny_224_1k.pth?download=true)|4scale | 12|55.1|47.8M|[model](https://huggingface.co/DaiShiResearch/dino-4scale-transnext-tiny-coco/resolve/main/dino_4scale_transnext_tiny_12e_in1k.pth?download=true)|[config](/detection/dino/configs/dino-4scale_transnext_tiny-12e_coco.py)|[log](https://huggingface.co/DaiShiResearch/dino-4scale-transnext-tiny-coco/raw/main/dino_4scale_transnext_tiny_12e_in1k.json)|
| TransNeXt-Tiny | [ImageNet-1K](https://huggingface.co/DaiShiResearch/transnext-tiny-224-1k/resolve/main/transnext_tiny_224_1k.pth?download=true)|5scale | 12|55.7|48.1M|[model](https://huggingface.co/DaiShiResearch/dino-5scale-transnext-tiny-coco/resolve/main/dino_5scale_transnext_tiny_12e_in1k.pth?download=true)|[config](/detection/dino/configs/dino-5scale_transnext_tiny-12e_coco.py)|[log](https://huggingface.co/DaiShiResearch/dino-5scale-transnext-tiny-coco/raw/main/dino_5scale_transnext_tiny_12e_in1k.json)|
| TransNeXt-Small | [ImageNet-1K](https://huggingface.co/DaiShiResearch/transnext-small-224-1k/resolve/main/transnext_small_224_1k.pth?download=true)|5scale | 12|56.6|69.6M|[model](https://huggingface.co/DaiShiResearch/dino-5scale-transnext-small-coco/resolve/main/dino_5scale_transnext_small_12e_in1k.pth?download=true)|[config](/detection/dino/configs/dino-5scale_transnext_small-12e_coco.py)|[log](https://huggingface.co/DaiShiResearch/dino-5scale-transnext-small-coco/raw/main/dino_5scale_transnext_small_12e_in1k.json)|
| TransNeXt-Base | [ImageNet-1K](https://huggingface.co/DaiShiResearch/transnext-base-224-1k/resolve/main/transnext_base_224_1k.pth?download=true)|5scale | 12|57.1|110M|[model](https://huggingface.co/DaiShiResearch/dino-5scale-transnext-base-coco/resolve/main/dino_5scale_transnext_base_12e_in1k.pth?download=true)|[config](/detection/dino/configs/dino-5scale_transnext_base-12e_coco.py)|[log](https://huggingface.co/DaiShiResearch/dino-5scale-transnext-base-coco/raw/main/dino_5scale_transnext_base_12e_in1k.json)|

## Requirements

    pip install -r requirements.txt

## Data preparation

    cd /path/to/current_folder
    ln -s /your/path/to/coco-dataset ./data

## Evaluation

To evaluate DINO models with TransNeXt backbone on COCO val, you can use the following command:

    bash dist_test.sh <config-file> <checkpoint-path> <gpu-num>

For example, to evaluate the TransNeXt-Tiny under 4-scale settings on a single GPU:

    bash dist_test.sh ./configs/dino-4scale_transnext_tiny-12e_coco.py /path/to/checkpoint_file 1

For example, to evaluate the TransNeXt-Tiny under 4-scale settings on 8 GPUs:

    bash dist_test.sh ./configs/dino-4scale_transnext_tiny-12e_coco.py /path/to/checkpoint_file 8

## Training

In order to train DINO models with TransNeXt backbone on the COCO dataset, first, you need to fill in the path of your
downloaded pretrained checkpoint in `./configs/<config-file>`. Specifically, change it to:

    pretrained=<path-to-checkpoint>, 

After setting up, to train TransNeXt on COCO dataset, you can use the following command:

    bash dist_train.sh <config-file> <gpu-num> 

For example, to train the TransNeXt-Tiny under 4-scale settings on 8 GPUs, with a total batch-size of 16:

    bash dist_train.sh ./configs/dino-4scale_transnext_tiny-12e_coco.py 8

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