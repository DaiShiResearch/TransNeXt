# UPerNet with TransNeXt backbone on ADE20K

## Model Zoo

**ADE20K semantic segmentation results using the UPerNet method:**

| Backbone | Pretrained Model| Crop Size |Lr Schd| mIoU|mIoU (ms+flip)| #Params | Download |Config| Log |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| TransNeXt-Tiny | [ImageNet-1K](https://huggingface.co/DaiShiResearch/transnext-tiny-224-1k/resolve/main/transnext_tiny_224_1k.pth?download=true)|512x512|160K|51.1|51.5/51.7|59M|[model](https://huggingface.co/DaiShiResearch/upernet-transnext-tiny-ade/resolve/main/upernet_transnext_tiny_512x512_160k_ade20k_in1k.pth?download=true)|[config](/segmentation/upernet/configs/upernet_transnext_tiny_512x512_160k_ade20k_ss.py)|[log](https://huggingface.co/DaiShiResearch/upernet-transnext-tiny-ade/blob/main/upernet_transnext_tiny_512x512_160k_ade20k_ss.log.json)|
| TransNeXt-Small | [ImageNet-1K](https://huggingface.co/DaiShiResearch/transnext-small-224-1k/resolve/main/transnext_small_224_1k.pth?download=true)|512x512|160K|52.2|52.5/52.8|80M|[model](https://huggingface.co/DaiShiResearch/upernet-transnext-small-ade/resolve/main/upernet_transnext_small_512x512_160k_ade20k_in1k.pth?download=true)|[config](/segmentation/upernet/configs/upernet_transnext_small_512x512_160k_ade20k_ss.py)|[log](https://huggingface.co/DaiShiResearch/upernet-transnext-small-ade/blob/main/upernet_transnext_small_512x512_160k_ade20k_ss.log.json)|
| TransNeXt-Base | [ImageNet-1K](https://huggingface.co/DaiShiResearch/transnext-base-224-1k/resolve/main/transnext_base_224_1k.pth?download=true)|512x512|160K|53.0|53.5/53.7|121M|[model](https://huggingface.co/DaiShiResearch/upernet-transnext-base-ade/resolve/main/upernet_transnext_base_512x512_160k_ade20k_in1k.pth?download=true)|[config](/segmentation/upernet/configs/upernet_transnext_base_512x512_160k_ade20k_ss.py)|[log](https://huggingface.co/DaiShiResearch/upernet-transnext-base-ade/blob/main/upernet_transnext_base_512x512_160k_ade20k_ss.log.json)|
* In the context of multi-scale evaluation, TransNeXt reports test results under two distinct scenarios: **interpolation** and **extrapolation** of relative position bias. 

## Requirements

    pip install -r requirements.txt

## Data preparation

    cd /path/to/current_folder
    ln -s /your/path/to/ade20k-dataset ./data

## Evaluation

***Single-scale Evaluation:***

To run single-scale evaluation of UPerNet models with TransNeXt backbone on ADE20K, you can use the following command:

    bash dist_test.sh <config-file-ending-with-"ss"> <checkpoint-path> <gpu-num> --eval mIoU

For example, to evaluate the TransNeXt-Tiny on a single GPU:
    
    bash dist_test.sh ./configs/upernet_transnext_tiny_512x512_160k_ade20k_ss.py /path/to/checkpoint_file 1 --eval mIoU
    
For example, to evaluate the TransNeXt-Tiny on 8 GPUs:
    
    bash dist_test.sh ./configs/upernet_transnext_tiny_512x512_160k_ade20k_ss.py /path/to/checkpoint_file 8 --eval mIoU

***Multi-scale Evaluation:***

To evaluate the pre-trained models with multi-scale inputs and flip augmentations on ADE20K under`interpolation of relative position bias` strategy, you can use the following command:
    
    bash dist_test.sh <config-file-ending-with-"ms"> <checkpoint-path> <gpu-num> --eval mIoU  --aug-test

You can also use the `<config-file-ending-with-"ms_extrapolation">` for multi-scale evaluation under `extrapolation of relative position bias` strategy, using the following command:

    bash dist_test.sh <config-file-ending-with-"ms_extrapolation"> <checkpoint-path> <gpu-num> --eval mIoU  --aug-test

## Training
In order to train UPerNet models with TransNeXt backbone on the ADE20K dataset, first, you need to fill in the path of your downloaded pretrained checkpoint in `./configs/<config-file-ending-with-"ss">`. Specifically, change it to:
    
    pretrained=<path-to-checkpoint>, 

After setting up, to train TransNeXt on ADE20K dataset, you can use the following command:
    
    bash dist_train.sh <config-file-ending-with-"ss"> <gpu-num> 

For example, to train the TransNeXt-Tiny on 8 GPUs, with a total batch-size of 16:

    bash dist_train.sh ./configs/upernet_transnext_tiny_512x512_160k_ade20k_ss.py 8

***Notice:** Our TransNeXt models are trained with single-scale images, if you want to reproduce accurately, please use `<config-file-ending-with-"ss">`.*

## Acknowledgement

The released script for Object Detection with TransNeXt is built based on the [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [timm](https://github.com/huggingface/pytorch-image-models) library.

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