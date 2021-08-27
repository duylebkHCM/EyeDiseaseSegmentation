
from collections import OrderedDict
from functools import partial
from typing import Union, List, Dict, Type

from pytorch_toolbelt.modules import conv1x1, UnetBlock, ACT_RELU, ABN, ACT_SWISH, Swish, ResidualDeconvolutionUpsample2d
from pytorch_toolbelt.modules import encoders as E
from pytorch_toolbelt.modules.decoders import UNetDecoder
from pytorch_toolbelt.modules.encoders import EncoderModule
from torch import nn, Tensor
from torch.nn import functional as F
from timm.models.efficientnet_blocks import InvertedResidual
from .modules import DropBlock2D

__all__ = [
    "UnetSegmentationModel",
    "resnet18_unet32",
    "resnet34_unet32",
    "resnet50_unet32",
    "resnet101_unet64",
    "resnet152_unet32",
    "densenet121_unet32",
    "densenet161_unet32",
    "densenet169_unet32",
    "densenet201_unet32",
    "b0_unet32_s2",
    "b4_unet32",
    "b4_effunet32",
    "b6_unet32_s2",
    "b6_unet32_s2_bi",
    "b6_unet32_s2_tc",
    "b6_unet32_s2_rdtc",
]

class UnetSegmentationModel(nn.Module):
    def __init__(
        self,
        encoder: EncoderModule,
        unet_channels: Union[int, List[int]],
        num_classes: int = 1,
        dropout=0.25,
        full_size_mask=True,
        activation=ACT_RELU,
        upsample_block: Union[Type[nn.Upsample], Type[ResidualDeconvolutionUpsample2d]] = nn.UpsamplingNearest2d,
        last_upsample_block=None,
        deep_supervision=False
    ):
        super().__init__()
        self.encoder = encoder

        abn_block = partial(ABN, activation=activation)
        self.decoder = UNetDecoder(
            feature_maps=encoder.channels,
            decoder_features=unet_channels,
            unet_block=partial(UnetBlock, abn_block=abn_block),
            upsample_block=upsample_block,
        )

        if last_upsample_block is not None:
            self.last_upsample_block = last_upsample_block(unet_channels[0])
            self.segmentation_head = nn.Sequential(
                nn.Dropout2d(dropout),
                conv1x1(self.last_upsample_block.out_channels, num_classes)
            )
        else:
            self.last_upsample_block = None

            self.segmentation_head = nn.Sequential(
                nn.Dropout2d(dropout),
                conv1x1(unet_channels[0], num_classes)
            )

        self.full_size_mask = full_size_mask
        if deep_supervision:
            self.supervision = nn.ModuleList([conv1x1(channels, num_classes) for channels in self.decoder.channels])
        else:
            self.supervision = None


    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x_size = x.size()
        x = self.encoder(x)
        x = self.decoder(x)

        # Decode mask
        if self.last_upsample_block is not None:
            mask = self.segmentation_head(self.last_upsample_block(x[0]))
        else:
            mask = self.segmentation_head(x[0])
            if self.full_size_mask:
                mask = F.interpolate(mask, size=x_size[2:], mode="bilinear", align_corners=False)
        
        prediction_list = []
        if self.supervision is not None:
            for feature_map, supervision  in zip(x, self.supervision):
                prediction = supervision(feature_map)
                prediction_list.append(prediction)
    
            return mask, prediction_list
        else:
            return mask


class EfficientUnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=Swish, drop_path_rate=0.0):
        super().__init__()
        self.ir = InvertedResidual(in_channels, out_channels, act_layer=activation, se_ratio=0.25, exp_ratio=4)
        self.drop = DropBlock2D(drop_path_rate, 2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation(inplace=True),
        )

    def forward(self, x):
        x = self.ir(x)
        x = self.drop(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class EfficientUNetDecoder(UNetDecoder):
    def __init__(
        self,
        feature_maps: List[int],
        decoder_features: List[int],
        upsample_block=nn.UpsamplingNearest2d,
        activation=Swish,
    ):
        super().__init__(
            feature_maps,
            unet_block=partial(EfficientUnetBlock, activation=activation, drop_path_rate=0.2),
            decoder_features=decoder_features,
            upsample_block=upsample_block,
        )

class EfficientUnetSegmentationModel(nn.Module):
    def __init__(
        self,
        encoder: EncoderModule,
        unet_channels: Union[int, List[int]],
        num_classes: int = 1,
        dropout=0.25,
        full_size_mask=True,
        activation=Swish,
    ):
        super().__init__()
        self.encoder = encoder

        self.decoder = EfficientUNetDecoder(
            feature_maps=encoder.channels, decoder_features=unet_channels, activation=activation
        )

        self.mask = nn.Sequential(
            nn.Dropout2d(dropout),
            conv1x1(self.decoder.channels[0], num_classes)
        )

        self.full_size_mask = full_size_mask

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x_size = x.size()
        enc = self.encoder(x)
        dec = self.decoder(enc)

        # Decode mask
        mask = self.mask(dec[0])

        if self.full_size_mask:
            mask = F.interpolate(mask, size=x_size[2:], mode="bilinear", align_corners=False)

        return mask


def resnet18_unet32(input_channels=3, num_classes=1, dropout=0.0, pretrained=True, deep_supervision=False):
    encoder = E.Resnet18Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(encoder, num_classes=num_classes, unet_channels=[32, 64, 128, 256], dropout=dropout, deep_supervision=deep_supervision)

def resnet34_unet32(input_channels=3, num_classes=1, dropout=0.0, pretrained=True, deep_supervision=False):
    encoder = E.Resnet34Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(encoder, num_classes=num_classes, unet_channels=[32, 64, 128, 256], dropout=dropout, deep_supervision=deep_supervision)

def resnet50_unet32(input_channels=3, num_classes=1, dropout=0.0, pretrained=True, deep_supervision=False):
    encoder = E.Resnet50Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(encoder, num_classes=num_classes, unet_channels=[32, 64, 128, 256], dropout=dropout, deep_supervision=deep_supervision)

def resnet101_unet64(input_channels=3, num_classes=1, dropout=0.5, pretrained=True, deep_supervision=False):
    encoder = E.Resnet101Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(encoder, num_classes=num_classes, unet_channels=[64, 128, 256, 512], dropout=dropout, deep_supervision=deep_supervision)

def resnet152_unet32(input_channels=3, num_classes=1, dropout=0.0, pretrained=True, deep_supervision=False):
    encoder = E.Resnet152Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(encoder, num_classes=num_classes, unet_channels=[32, 64, 128, 256], dropout=dropout, deep_supervision=deep_supervision)

def densenet121_unet32(input_channels=3, num_classes=1, dropout=0.0, pretrained=True, deep_supervision=False):
    encoder = E.DenseNet121Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(encoder, num_classes=num_classes, unet_channels=[32, 64, 128, 256], dropout=dropout, deep_supervision=deep_supervision)

def densenet161_unet32(input_channels=3, num_classes=1, dropout=0.0, pretrained=True, deep_supervision=False):
    encoder = E.DenseNet161Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(encoder, num_classes=num_classes, unet_channels=[32, 64, 128, 256], dropout=dropout, deep_supervision=deep_supervision)

def densenet169_unet32(input_channels=3, num_classes=1, dropout=0.0, pretrained=True, deep_supervision=False):
    encoder = E.DenseNet169Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(encoder, num_classes=num_classes, unet_channels=[32, 64, 128, 256], dropout=dropout, deep_supervision=deep_supervision)

def b0_unet32_s2(input_channels=3, num_classes=1, dropout=0.1, pretrained=True):
    encoder = E.B0Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(
        encoder, num_classes=num_classes, unet_channels=[16, 32, 64, 128], activation=ACT_SWISH, dropout=dropout
    )

def b4_unet32(input_channels=3, num_classes=1, dropout=0.2, pretrained=True):
    encoder = E.B4Encoder(pretrained=pretrained)
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return UnetSegmentationModel(
        encoder, num_classes=num_classes, unet_channels=[32, 64, 128], activation=ACT_SWISH, dropout=dropout
    )

def b4_effunet32(input_channels=3, num_classes=1, dropout=0.2, pretrained=True):
    encoder = E.B4Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_inencput_channels(input_channels)

    return EfficientUnetSegmentationModel(
        encoder, num_classes=num_classes, unet_channels=[32, 64, 128, 256], activation=Swish, dropout=dropout
    )

def b2_effunet32(input_channels=3, num_classes=1, dropout=0.2, pretrained=True):
    encoder = E.B2Encoder(pretrained=pretrained, layers=[0, 1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return EfficientUnetSegmentationModel(
        encoder, num_classes=num_classes, unet_channels=[32, 64, 128, 256], activation=Swish, dropout=dropout
    )

if __name__ == '__main__':
    import torch
    a  = torch.randn(2, 3, 1024, 1024)
    model = resnet18_unet32(deep_supervision=True)

    output, deep_output = model(a)
    print(output.shape)
    for d in deep_output:
        print(d.shape)
