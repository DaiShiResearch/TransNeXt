# ImageNet-1K classification with TransNeXt

## Model Zoo

**ImageNet-1K 224x224 pre-trained models:**

| Model | #Params | #FLOPs |IN-1K | IN-A | IN-C&#8595; |IN-R|Sketch|IN-V2|Download |Config| Log |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|:---:| :---:|:---:|:---:|:---:|
| TransNeXt-Micro|12.8M|2.7G| 82.5 | 29.9 | 50.8|45.8|33.0|72.6|[model](https://huggingface.co/DaiShiResearch/transnext-micro-224-1k/resolve/main/transnext_micro_224_1k.pth?download=true) |[config](/classification/configs/transnext_micro.py)|[log](https://huggingface.co/DaiShiResearch/transnext-micro-224-1k/raw/main/transnext_micro_224_1k.txt) |
| TransNeXt-Tiny |28.2M|5.7G| 84.0| 39.9| 46.5|49.6|37.6|73.8|[model](https://huggingface.co/DaiShiResearch/transnext-tiny-224-1k/resolve/main/transnext_tiny_224_1k.pth?download=true)|[config](/classification/configs/transnext_tiny.py)|[log](https://huggingface.co/DaiShiResearch/transnext-tiny-224-1k/raw/main/transnext_tiny_224_1k.txt)|
| TransNeXt-Small |49.7M|10.3G| 84.7| 47.1| 43.9|52.5| 39.7|74.8 |[model](https://huggingface.co/DaiShiResearch/transnext-small-224-1k/resolve/main/transnext_small_224_1k.pth?download=true)|[config](/classification/configs/transnext_small.py)|[log](https://huggingface.co/DaiShiResearch/transnext-small-224-1k/raw/main/transnext_small_224_1k.txt)|
| TransNeXt-Base |89.7M|18.4G| 84.8| 50.6|43.5|53.9|41.4|75.1| [model](https://huggingface.co/DaiShiResearch/transnext-base-224-1k/resolve/main/transnext_base_224_1k.pth?download=true)|[config](/classification/configs/transnext_base.py)|[log](https://huggingface.co/DaiShiResearch/transnext-base-224-1k/raw/main/transnext_base_224_1k.txt)|

**ImageNet-1K 384x384 fine-tuned models:**

| Model | #Params | #FLOPs |IN-1K | IN-A |IN-R|Sketch|IN-V2| Download |Config| 
|:---:|:---:|:---:|:---:| :---:|:---:|:---:| :---:|:---:|:---:|
| TransNeXt-Small |49.7M|32.1G| 86.0| 58.3|56.4|43.2|76.8| [model](https://huggingface.co/DaiShiResearch/transnext-small-384-1k-ft-1k/resolve/main/transnext_small_384_1k_ft_1k.pth?download=true)|[config](/classification/configs/finetune/transnext_small_384_ft.py)|
| TransNeXt-Base |89.7M|56.3G| 86.2| 61.6|57.7|44.7|77.0| [model](https://huggingface.co/DaiShiResearch/transnext-base-384-1k-ft-1k/resolve/main/transnext_base_384_1k_ft_1k.pth?download=true)|[config](/classification/configs/finetune/transnext_base_384_ft.py)|

**ImageNet-1K 256x256 pre-trained model fully utilizing aggregated attention at all stages:**

*(See Table.9 in Appendix D.6 for details)*

| Model |Token mixer| #Params | #FLOPs |IN-1K |Download |Config| Log |
|:---:|:---:|:---:|:---:| :---:|:---:|:---:|:---:|
|TransNeXt-Micro|**A-A-A-A**|13.1M|3.3G| 82.6 |[model](https://huggingface.co/DaiShiResearch/transnext-micro-AAAA-256-1k/resolve/main/transnext_micro_AAAA_256_1k.pth?download=true) |[config](/classification/configs/transnext_micro_AAAA_256.py)|[log](https://huggingface.co/DaiShiResearch/transnext-micro-AAAA-256-1k/blob/main/transnext_micro_AAAA_256_1k.txt) |

***Notice:** Unlike [ConvNeXt](https://github.com/facebookresearch/ConvNeXt), we did not use the weight EMA techniques, and the
pre-training only selects the best checkpoint during training, following the practice
of [BiFormer](https://github.com/rayleizhu/BiFormer) and other works. Specific details can be referred to the training
log.*

## Requirements

    pip install -r requirements.txt

## Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html), and the training and validation data is expected to be in the `train` folder and `val` folder respectively:

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

## Evaluation

To evaluate the pre-trained TransNeXt-Base on ImageNet-1K val with a single GPU:

```
bash dist_train.sh ./configs/transnext_base.py 1 --data-path /path/to/imagenet --resume /path/to/checkpoint_file --eval
```

This should give (We tested on T4):

```
* Acc@1 84.845 Acc@5 97.092 loss 0.670
Accuracy of the network on the 50000 test images: 84.8%
```

The more general command format is as follows, which is also used for the fine-tuned model with a resolution of 384x384
and the variant model that fully uses aggregated attention:

```
bash dist_train.sh <config-file> 1 --data-path <path-to-imagenet> --resume <path-to-checkpoint> --eval
```

To perform multi-scale input inference, you can use the following command:

```
bash dist_train.sh <config-file> 1 --data-path <path-to-imagenet> --input-size <input-size> --pretrain-size 224 --resume <path-to-checkpoint> --eval
```
* *The `pretrain-size` is only related to the pre-training stage. For the model pretrained at a resolution of 224x224, as well as those models subsequently fine-tuned at 384x384, you should use `--pretrain-size 224`. Similarly, for the `TransNeXt-Micro-AAAA` pretrained at a resolution of 256x256, it should be set to 256.*

To perform multi-scale input inference in the `linear inference mode` of TransNeXt, you can add `--fixed-pool-size <pool-size>` at the end of the command, as follows:

```
bash dist_train.sh <config-file> 1 --data-path <path-to-imagenet> --input-size <input-size> --pretrain-size 224 --fixed-pool-size <pool-size> --resume <path-to-checkpoint> --eval
```

* *In our paper, when testing the `linear inference mode` of the model pretrained at a resolution of 224x224, we set `<pool-size>` to 7. You can also try other values to achieve different performance-efficiency trade-offs.*

***Notice:** Please note that the current script does not support multi-GPU evaluation, using more than one GPU for evaluation will
not speed up.*

## ImageNet-1K Training

To train TransNeXt-Micro on ImageNet-1K using 8 GPUs:

```
bash dist_train.sh ./configs/transnext_micro.py 8 --data-path /path/to/imagenet --batch-size 128
```

To train TransNeXt-Micro on ImageNet-1K using 2 GPUs and 4 steps of gradient accumulation for the same batch-size training:


```
bash dist_train.sh ./configs/transnext_micro.py 2 --data-path /path/to/imagenet --batch-size 128 --update-freq 4
```

Please note that we use a batch size of 1024 for training in the paper. If you want to accurately reproduce, please make sure that `<num-of-gpus> * <batch-size-per-gpu> * <steps-of-gradient-accumulation> = 1024`, as follows:

```
bash dist_train.sh <config-file> <num-of-gpus> --data-path /path/to/imagenet --batch-size <batch-size-per-gpu> --update-freq <steps-of-gradient-accumulation>
```

## ImageNet-1K Fine-tuning

To fine-tune the TransNeXt-Small model pre-trained at a resolution of `224x224` on 8 GPUs at a resolution of `384x384`:

```
bash dist_train.sh ./configs/finetune/transnext_small_384_ft.py 8 --data-path /path/to/imagenet --batch-size 128  --resume <path-to-pretrained-weights>
```
The command format is similar to ImageNet-1K pre-training, you only need to use the configuration file in the `./configs/finetune/` directory and the pre-trained weights downloaded from the model zoo to start training.

## Support for torch.compile mode
This training script supports the torch.compile mode and uses this mode by default for evaluation/training. Torch.compile will compile the model at the start of the run, which may take a long time. If you do not want to wait, you can add `--no-compile-model` at the end of the above command to switch back to PyTorch's standard eager mode.

***Notice:** Since the two modes use different kernels, it is normal for the same model to have very slight
differences in calculation results under the two modes.*

## Acknowledgement

The released PyTorch ImageNet training script is based on the code of [PVT](https://github.com/whai362/PVT) repositories, which was built using the [timm](https://github.com/huggingface/pytorch-image-models) library, [DeiT](https://github.com/facebookresearch/deit) repositories.

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
