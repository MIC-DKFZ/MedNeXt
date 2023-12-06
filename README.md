# MedNeXt

Copyright © German Cancer Research Center (DKFZ), [Division of Medical Image Computing (MIC)](https://www.dkfz.de/en/mic/index.php). Please make sure that your usage of this code is in compliance with the [code license](https://github.com/MIC-DKFZ/MedNeXt/blob/main/LICENSE). 

**MedNeXt** is a fully ConvNeXt architecture for 3D medical image segmentation designed to leverage the 
scalability of the ConvNeXt block while being customized to the challenges of sparsely annotated medical 
image segmentation datasets. MedNeXt is a model under development and is expected to be updated 
periodically in the near future. 

The current training framework is built on top of nnUNet (v1) - the module name `nnunet_mednext` reflects this. You are free to adopt the architecture for your own training pipeline or use the one in this repository. Instructions are provided for both paths. 

[**[arXiv, 2023]**](https://arxiv.org/abs/2303.09975) [**[MICCAI 2023]**](https://link.springer.com/chapter/10.1007/978-3-031-43901-8_39)

Please cite the following work if you find this model useful for your research:

    Roy, S., Koehler, G., Ulrich, C., Baumgartner, M., Petersen, J., Isensee, F., Jaeger, P.F. & Maier-Hein, K. (2023).
    MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation. 
    International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI), 2023.

Please also cite the following work if you use this pipeline for training:

    Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2020). 
    nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 1-9.

# Table of Contents
- [MedNeXt](#mednext)
- [Table of Contents](#table-of-contents)
- [Current version and notable features](#current-versions-and-notable-features)
- [Installation](#installation)
- [MedNeXt Architecture and Usage in external pipelines](#mednext-architecture-and-usage-in-external-pipelines)
  - [MedNeXt v1](#mednext-v1)
    - [Usage as whole MedNeXt v1 architecture](#usage-as-whole-mednext-v1-architecture)
    - [Individual Usage of MedNeXt blocks](#individual-usage-of-mednext-blocks)
    - [UpKern weight loading](#upkern-weight-loading)
- [Usage of internal training pipeline](#usage-of-internal-training-pipeline)
    - [Plan and Preprocess](#plan-and-preprocess)
    - [Train MedNeXt using nnUNet (v1) training](#train-mednext-using-nnunet-v1-training)
    - [Train a kernel 5x5x5 version using UpKern](#train-a-kernel-5x5x5-version-using-upkern)

# Current Versions and notable features:

- **v1 (MICCAI 2023)**: Fully 3D ConvNeXt architecture, residual ConvNeXt resampling, UpKern for large kernels, gradient checkpointing for training large models

As mentioned earlier, MedNeXt is actively under development and further improvements to the pipeline as future versions are anticipated.

# Installation
The repository can be cloned and installed using the following commands.

```bash
git clone https://github.com/MIC-DKFZ/MedNeXt.git mednext
cd mednext
pip install -e .
```

# MedNeXt Architecture and Usage in external pipelines
MedNeXt is usable on external training pipeline for 3D volumetric segmentation, similar to any PyTorch `nn.Module`. It is functionally decoupled from nnUNet when used simply as an architecture. It is sufficient to install the repository and import either the architecture or the block. In theory, it is possible to freely customize the network using MedNeXt both as an encoder-decoder style network as well as a block.

## MedNeXt v1 

MedNeXt v1 is the first version of the MedNeXt and incorporates the architectural features described [here](#current-versions-and-notable-features). 

**_Important_:** MedNeXt v1 was trained with *1.0mm isotropic spacing* as favored by architectures like [UNETR](https://arxiv.org/abs/2103.10504), [SwinUNETR](https://arxiv.org/abs/2201.01266) and the usage of alternate spacing, like *median spacing* favored by native nnUNet, while perfectly usable in theory, is currently untested with MedNeXt v1 and may affect performance.

The usage as whole MedNeXt v1 as a complete architecture as well as the use of MedNeXt blocks (in external architectures, for example) is described below.

### Usage as whole MedNeXt v1 architecture:

The architecture can be imported as follows with a number of arguments.

```
from nnunet_mednext.mednextv1 import MedNeXt

model = MedNeXt(
          in_channels: int,                         # input channels
          n_channels: int,                          # number of base channels
          n_classes: int,                           # number of classes
          exp_r: int = 4,                           # Expansion ratio in Expansion Layer
          kernel_size: int = 7,                     # Kernel Size in Depthwise Conv. Layer
          enc_kernel_size: int = None,              # (Separate) Kernel Size in Encoder
          dec_kernel_size: int = None,              # (Separate) Kernel Size in Decoder
          deep_supervision: bool = False,           # Enable Deep Supervision
          do_res: bool = False,                     # Residual connection in MedNeXt block
          do_res_up_down: bool = False,             # Residual conn. in Resampling blocks
          checkpoint_style: bool = None,            # Enable Gradient Checkpointing
          block_counts: list = [2,2,2,2,2,2,2,2,2], # Depth-first no. of blocks per layer 
          norm_type = 'group',                      # Type of Norm: 'group' or 'layer'
          dim = '3d'                                # Supports `3d', '2d' arguments
)
```

Please note that - 1) Deep Supervision, and 2) residual connections in both MedNeXt and Up/Downsampling blocks are both used in the publication for training. 

Gradient Checkpointing can be used to train larger models in low memory devices by trading compute for activation storage. The checkpointing implemented in this version is at the MedNeXt block level.

MedNeXt v1  has been tested with 4 defined architecture sizes and 2 defined kernel sizes. Their particulars are as follows:

| Name (Model ID) | Kernel Size | Parameters | GFlops |
|-----|---- |-----| -----|
| Small (S) | 3x3x3 | 5.6M | 130 | 
| Small (S) | 5x5x5 | 5.9M | 169 |
| Base (B) | 3x3x3 | 10.5M | 170 |
| Base (B) | 5x5x5 | 11.0M | 208 |
| Medium (M) | 3x3x3 | 17.6M | 248 |
| Medium (M) | 5x5x5 | 18.3M | 308 |
| Large (L) | 3x3x3 | 61.8M | 500 |
| Large (L) | 5x5x5 | 63.0M | 564 |

Utility functions have been defined for re-creating these architectures (with or without deep supervision) as follows customized to input channels, number of target classes, model IDs as used in the publication, kernel size and deep supervision:

```
from nnunet_mednext import create_mednext_v1

model = create_mednext_v1(
  num_channels = 3,
  num_classes = 10,
  model_id = 'B',             # S, B, M and L are valid model ids
  kernel_size = 3,            # 3x3x3 and 5x5x5 were tested in publication
  deep_supervision = True     # was used in publication
)
```

### Individual Usage of MedNeXt blocks
MedNeXt blocks can be imported for use individually similar to the entire architecture. The following blocks can be imported directed for use.

```
from nnunet_mednext import MedNeXtBlock, MedNeXtDownBlock, MedNeXtUpBlock

# Standard MedNeXt block
block = MedNeXtBlock(
    in_channels:int,              # no. of input channels
    out_channels:int,             # no. of output channels
    exp_r:int=4,                  # channel expansion ratio in Expansion Layer
    kernel_size:int=7,            # kernel size in Depthwise Conv. Layer
    do_res:bool=True,              # residual connection on or off. Default: True
    norm_type:str = 'group',      # type of norm: 'group' or 'layer'
    n_groups:int or None = None,  # no. of groups in Depthwise Conv. Layer
                                  # (keep 'None' in most cases)
)


# 2x Downsampling with MedNeXt block
block_down = MedNeXtDownBlock(
    in_channels:int,              # no. of input channels
    out_channels:int,             # no. of output channels
    exp_r:int=4,                  # channel expansion ratio in Expansion Layer
    kernel_size:int=7,            # kernel size in Depthwise Conv. Layer
    do_res:bool=True,              # residual connection on or off. Default: True
    norm_type:str = 'group',      # type of norm: 'group' or 'layer'
)


# 2x Upsampling with MedNeXt block
block_up = MedNeXtUpBlock(
    in_channels:int,              # no. of input channels
    out_channels:int,             # no. of output channels
    exp_r:int=4,                  # channel expansion ratio in Expansion Layer
    kernel_size:int=7,            # kernel size in Depthwise Conv. Layer
    do_res:bool=True,              # residual connection on or off. Default: True
    norm_type:str = 'group',      # type of norm: 'group' or 'layer'
)
```

### UpKern weight loading

UpKern is a simple algorithm for initializing a large kernel MedNeXt network with an *equivalent* small kernel MedNeXt. Equivalent refers to a network of the same configuration with the *only* difference being kernel size in the Depthwise Convolution layers. 
Large kernels are initialized by trilinear interpolation of their smaller counterparts.
The following is an example of using this weight loading style.

```
from nnunet_mednext import create_mednext_v1
from nnunet_mednext.run.load_weights import upkern_load_weights
m_net_ = create_mednext_v1(1, 3, 'S', 5)
m_pre = create_mednext_v1(1, 3, 'S', 3)

# Generally m_pre would be pretrained
m3 = upkern_load_weights(m_net_, m_pre)
```

# Usage of internal training pipeline

## Plan and Preprocess

To preprocess your datasets as in the MICCAI 2023 version, please run

```
mednextv1_plan_and_preprocess -t YOUR_TASK -pl3d ExperimentPlanner3D_v21_customTargetSpacing_1x1x1 -pl2d ExperimentPlanner2D_v21_customTargetSpacing_1x1x1
```

As in nnUNet, you can set `-pl3d` or `-pl2d` as `None` if you do not require preprocessed data in those dimensions.
Please note that `YOUR_TASK` in this repo is designed to be in the old nnUNet(v1) format. If you want to use the 
latest nnUNet (v2), you will have to adopt the preprocessor on your own.

The custom `ExperimentPlanner3D_v21_customTargetSpacing_1x1x1` is designed to set patch size to `128x128x128` and
spacing to 1mm isotropic since those are the experimental conditions used in the MICCAI 2023 version. 

## Train MedNeXt using nnUNet (v1) training
MedNeXt has custom nnUNet (v1) trainers that allow it to be trained similar to the base architecture. 
Please check the old nnUNet(v1) branch in the nnUNet repo, if you are unfamiliar with this code format. Please look [here](https://github.com/MIC-DKFZ/MedNeXt/blob/main/nnunet_mednext/training/network_training/MedNeXt/nnUNetTrainerV2_MedNeXt.py) for all available trainers to recreate the MICCAI 2023 experiments. Please note that all trainers are in 3D since the architecture was tested in 3D. You can of course, create your custom trainers if you want (including 2D trainers for 2D architectures). 
```
mednextv1_train 3d_fullres TRAINER TASK_NUMBER FOLD -p nnUNetPlansv2.1_trgSp_1x1x1
```

There are trainers for 4 architectures (`S`, `B`, `M`, `L`) and 2 kernel sizes (3, 5) to replicate the experiments from MICCAI 2023.
The following is an example for training an `nnUNetTrainerV2_MedNeXt_S_kernel3` trainer on the task `Task040_KiTS2019` on fold 0
of the 5-folds split generated by nnUNet's data preprocessor.

```
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_S_kernel3 Task040_KiTS2019 0 -p nnUNetPlansv2.1_trgSp_1x1x1
```

A kernel `5x5x5` version from scratch can also be trained this way, although we recommend initially training a kernel `3x3x3` version
and using UpKern.

## Train a kernel 5x5x5 version using UpKern
To train a kernel `5x5x5` version using UpKern, a kernel `3x3x3` version must already be trained. To train using UpKern, simply run the 
following:

```
mednextv1_train 3d_fullres TRAINER TASK FOLD -p nnUNetPlansv2.1_trgSp_1x1x1 -pretrained_weights YOUR_MODEL_CHECKPOINT_FOR_KERNEL_3_FOR_SAME_TASK_AND_FOLD -resample_weights
```

The following is an example for training an `nnUNetTrainerV2_MedNeXt_S_kernel5` trainer on the task `Task040_KiTS2019` on fold 0 of the 5-folds split generated by nnUNet's data preprocessor by using UpKern.

```
mednextv1_train 3d_fullres nnUNetTrainerV2_MedNeXt_S_kernel5 Task040_KiTS2019 0 -p nnUNetPlansv2.1_trgSp_1x1x1 -pretrained_weights SOME_PATH/nnUNet/3d_fullres/Task040_KiTS2019/nnUNetTrainerV2_MedNeXt_S_kernel3__nnUNetPlansv2.1_trgSp_1x1x1/fold_0/model_final_checkpoint.model -resample_weights
```

The `-resample_weights` flag as it is responsible to triggering the UpKern algorithm.

### A note on 2D MedNeXt:
Please note that while the MedNeXt can run on 2D, it has not been tested in 2D mode.