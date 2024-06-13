# Mask2Former with TransNeXt backbone on ADE20K

## Model Zoo

**ADE20K semantic segmentation results using the Mask2Former method:**

| Backbone | Pretrained Model| Crop Size |Lr Schd| mIoU| #Params | Download |Config| Log |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| TransNeXt-Tiny | [ImageNet-1K](https://huggingface.co/DaiShiResearch/transnext-tiny-224-1k/resolve/main/transnext_tiny_224_1k.pth?download=true)|512x512|160K|53.4|47.5M|[model](https://huggingface.co/DaiShiResearch/mask2former-transnext-tiny-ade/resolve/main/mask2former_transnext_tiny_512x512_160k_ade20k_in1k.pth?download=true)|[config](/segmentation/mask2former/configs/mask2former_transnext_tiny_160k_ade20k-512x512.py)|[log](https://huggingface.co/DaiShiResearch/mask2former-transnext-tiny-ade/raw/main/mask2former_transnext_tiny_512x512_160k_ade20k_in1k.json)|
| TransNeXt-Small | [ImageNet-1K](https://huggingface.co/DaiShiResearch/transnext-small-224-1k/resolve/main/transnext_small_224_1k.pth?download=true)|512x512|160K|54.1|69.0M|[model](https://huggingface.co/DaiShiResearch/mask2former-transnext-small-ade/resolve/main/mask2former_transnext_small_512x512_160k_ade20k_in1k.pth?download=true)|[config](/segmentation/mask2former/configs/mask2former_transnext_small_160k_ade20k-512x512.py)|[log](https://huggingface.co/DaiShiResearch/mask2former-transnext-small-ade/raw/main/mask2former_transnext_small_512x512_160k_ade20k_in1k.json)|
| TransNeXt-Base | [ImageNet-1K](https://huggingface.co/DaiShiResearch/transnext-base-224-1k/resolve/main/transnext_base_224_1k.pth?download=true)|512x512|160K|54.7|109M|[model](https://huggingface.co/DaiShiResearch/mask2former-transnext-base-ade/resolve/main/mask2former_transnext_base_512x512_160k_ade20k_in1k.pth?download=true)|[config](/segmentation/mask2former/configs/mask2former_transnext_base_160k_ade20k-512x512.py)|[log](https://huggingface.co/DaiShiResearch/mask2former-transnext-base-ade/raw/main/mask2former_transnext_base_512x512_160k_ade20k_in1k.json)|

## Requirements

    pip install -r requirements.txt

## Data preparation

    cd /path/to/current_folder
    ln -s /your/path/to/ade20k-dataset ./data

## Evaluation

To evaluate Mask2Former models with TransNeXt backbone on ADE20K val, you can use the following command:

    bash dist_test.sh <config-file> <checkpoint-path> <gpu-num>

For example, to evaluate the TransNeXt-Tiny under 4-scale settings on a single GPU:

    bash dist_test.sh ./configs/mask2former_transnext_tiny_160k_ade20k-512x512.py /path/to/checkpoint_file 1

For example, to evaluate the TransNeXt-Tiny under 4-scale settings on 8 GPUs:

    bash dist_test.sh ./configs/mask2former_transnext_tiny_160k_ade20k-512x512.py /path/to/checkpoint_file 8

## Training

In order to train Mask2Former models with TransNeXt backbone on the ADE20K dataset, first, you need to fill in the path of your
downloaded pretrained checkpoint in `./configs/<config-file>`. Specifically, change it to:

    pretrained=<path-to-checkpoint>, 

After setting up, to train TransNeXt on ADE20K dataset, you can use the following command:

    bash dist_train.sh <config-file> <gpu-num> 

For example, to train the TransNeXt-Tiny under 4-scale settings on 8 GPUs, with a total batch-size of 16:

    bash dist_train.sh ./configs/mask2former_transnext_tiny_160k_ade20k-512x512.py 8

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