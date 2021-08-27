from pytorch_toolbelt.modules.backbone.senet import se_resnet50
import torch
from torch import nn
from torch.nn import functional as F
from segmentation_models_pytorch.base import modules as md
from typing import Optional, Union, List
from segmentation_models_pytorch.deeplabv3.decoder import DeepLabV3PlusDecoder
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import SegmentationModel
from segmentation_models_pytorch.base import SegmentationHead, ClassificationHead
from .model_util import get_lr_parameters
from segmentation_models_pytorch.base import initialization as init


class DeepLabV3Plus(SegmentationModel):
    """DeepLabV3+ implemetation from "Encoder-Decoder with Atrous Separable
    Convolution for Semantic Image Segmentation"
    
    Args:
        encoder_name: Name of the classification model that will be used as an encoder (a.k.a backbone)
            to extract features of different spatial resolution
        encoder_depth: A number of stages used in encoder in range [3, 5]. Each stage generate features 
            two times smaller in spatial dimentions than previous one (e.g. for depth 0 we will have features 
            with shapes [(N, C, H, W),], for depth 1 - [(N, C, H, W), (N, C, H // 2, W // 2)] and so on).
            Default is 5
        encoder_weights: One of **None** (random initialization), **"imagenet"** (pre-training on ImageNet) and 
            other pretrained weights (see table with available weights for each encoder_name)
        encoder_output_stride: Downsampling factor for last encoder features (see original paper for explanation)
        decoder_atrous_rates: Dilation rates for ASPP module (should be a tuple of 3 integer values)
        decoder_channels: A number of convolution filters in ASPP module. Default is 256
        in_channels: A number of input channels for the model, default is 3 (RGB images)
        classes: A number of classes for output mask (or you can think as a number of channels of output mask)
        activation: An activation function to apply after the final convolution layer.
            Avaliable options are **"sigmoid"**, **"softmax"**, **"logsoftmax"**, **"identity"**, **callable** and **None**.
            Default is **None**
        upsampling: Final upsampling factor. Default is 4 to preserve input-output spatial shape identity
        aux_params: Dictionary with parameters of the auxiliary output (classification head). Auxiliary output is build 
            on top of encoder if **aux_params** is not **None** (default). Supported params:
                - classes (int): A number of classes
                - pooling (str): One of "max", "avg". Default is "avg"
                - dropout (float): Dropout factor in [0, 1)
                - activation (str): An activation function to apply "sigmoid"/"softmax" (could be **None** to return logits)
    Returns:
        ``torch.nn.Module``: **DeepLabV3Plus**
    
    Reference:
        https://arxiv.org/abs/1802.02611v3

    """
    def __init__(
            self,
            encoder_name: str = "resnet34",
            encoder_depth: int = 5,
            encoder_weights: Optional[str] = "imagenet",
            encoder_output_stride: int = 16,
            decoder_channels: int = 256,
            decoder_atrous_rates: tuple = (12, 24, 36),
            in_channels: int = 3,
            classes: int = 1,
            activation: Optional[str] = None,
            upsampling: int = 4,
            aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )
        
        if encoder_weights is not None:
            bn_types = nn.BatchNorm2d
            #Disable batchnorm update
            for m in self.encoder.modules():
                if isinstance(m, bn_types):
                    m.eval()

        if encoder_output_stride == 8:
            self.encoder.make_dilated(
                stage_list=[4, 5],
                dilation_list=[2, 4]
            )

        elif encoder_output_stride == 16:
            self.encoder.make_dilated(
                stage_list=[5],
                dilation_list=[2]
            )
        else:
            raise ValueError(
                "Encoder output stride should be 8 or 16, got {}".format(encoder_output_stride)
            )

        self.decoder = DeepLabV3PlusDecoder(
            encoder_channels=self.encoder.out_channels,
            out_channels=decoder_channels,
            atrous_rates=decoder_atrous_rates,
            output_stride=encoder_output_stride,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=self.decoder.out_channels,
            out_channels=classes,
            activation=activation,
            kernel_size=1,
            upsampling=upsampling,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

    def get_num_parameters(self):
        trainable= int(sum(p.numel() for p in self.parameters() if p.requires_grad))
        total = int(sum(p.numel() for p in self.parameters()))
        return trainable, total
    
    def get_paramgroup(self, base_lr=None, weight_decay=1e-5):
        lr_dict = {
            "encoder": [0.1, weight_decay]
        }
        
        lr_group = get_lr_parameters(self, base_lr, lr_dict)
        return lr_group

