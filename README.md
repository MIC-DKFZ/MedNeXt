# MedNeXt

Copyright © German Cancer Research Center (DKFZ), [Division of Medical Image Computing (MIC)](https://www.dkfz.de/en/mic/index.php). Please make sure that your usage of this code is in compliance with the [code license](https://github.com/MIC-DKFZ/MedNeXt/blob/main/LICENSE). 

**MedNeXt** is a fully ConvNeXt architecture for 3D medical image segmentation designed to leverage the 
scalability of the ConvNeXt block while being customized to the challenges of sparsely annotated medical 
image segmentation datasets. MedNeXt is a model under development and is expected to be updated 
periodically in the near future. 

The current training framework is built on top of nnUNet (v1) - the module name `nnunet_mednext` reflects this. You are free to adopt the architecture for your own training pipeline or use the one in this repository. Instructions are provided for both paths. 

Please cite the following work if you find this model useful for your research:

    Roy, S., Koehler, G., Ulrich, C., Baumgartner, M., Petersen, J., Isensee, F., Jaeger, P.F. & Maier-Hein, K.(2023). 
    MedNeXt: Transformer-driven Scaling of ConvNets for Medical Image Segmentation. arXiv preprint arXiv:2303.09975.

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

# Current Versions and notable features:

- **v1 (MICCAI 2023)**: Fully 3D ConvNeXt architecture, residual ConvNeXt resampling, UpKern for large kernels, gradient checkpointing for training large models

As mentioned earlier, MedNeXt is actively under development and further improvements to the pipeline as future versions are anticipated.

# Installation
The repository can be cloned and installed using the following commands.

```bash
git clone https://github.com/MIC-DKFZ/MedNeXt.git mednext
cd mednext
pip install -U .
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
          norm_type = 'group'                       # Type of Norm: 'group' or 'layer'
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
    do_res:int=True,              # residual connection on or off. Default: True
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
    do_res:int=True,              # residual connection on or off. Default: True
    norm_type:str = 'group',      # type of norm: 'group' or 'layer'
)


# 2x Upsampling with MedNeXt block
block_up = MedNeXtUpBlock(
    in_channels:int,              # no. of input channels
    out_channels:int,             # no. of output channels
    exp_r:int=4,                  # channel expansion ratio in Expansion Layer
    kernel_size:int=7,            # kernel size in Depthwise Conv. Layer
    do_res:int=True,              # residual connection on or off. Default: True
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

# Generally m2 would be pretrained
m3 = upkern_load_weights(m_net_, m_pre)
```

# Usage of internal training pipeline

**To Be Added Soon**