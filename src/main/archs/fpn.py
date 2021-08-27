from collections import OrderedDict
from functools import partial

from pytorch_toolbelt.modules import ABN, conv1x1, ACT_RELU, FPNContextBlock, FPNBottleneckBlock, ACT_SWISH, FPNFuse
from pytorch_toolbelt.modules import encoders as E
from pytorch_toolbelt.modules.decoders import FPNSumDecoder, FPNCatDecoder
from pytorch_toolbelt.modules.encoders import EncoderModule
from torch import nn
from torch.nn import functional as F
from pytorch_toolbelt.modules.encoders.timm import B4Encoder, B2Encoder
from .model_util import get_lr_parameters

__all__ = [
    "FPNSumSegmentationModel",
    "FPNCatSegmentationModel",
    "resnet34_fpncat128",
    "resnet152_fpncat256",
    "seresnext50_fpncat128",
    "seresnext101_fpncat256",
    "seresnext101_fpnsum256",
    "effnetB4_fpncat128",
]

class FPNSumSegmentationModel(nn.Module):
    def __init__(
        self, encoder: EncoderModule, num_classes: int, dropout=0.25, full_size_mask=True, fpn_channels=256, deep_supervision=False
    ):
        super().__init__()
        self.encoder = encoder

        self.decoder = FPNSumDecoder(feature_maps=encoder.output_filters, fpn_channels=fpn_channels,)
        self.mask = nn.Sequential(
            OrderedDict([("drop", nn.Dropout2d(dropout)), ("conv", conv1x1(fpn_channels, num_classes))])
        )

        self.full_size_mask = full_size_mask
        if deep_supervision:
            self.supervision = nn.ModuleList([conv1x1(channels, num_classes) for channels in self.decoder.channels])
        else:
            self.supervision = None


    def forward(self, x):
        x_size = x.size()
        x = self.encoder(x)
        x = self.decoder(x)

        # Decode mask
        mask = self.mask(x[0])

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

class FPNCatSegmentationModel(nn.Module):
    def __init__(
        self,
        encoder: EncoderModule,
        num_classes: int,
        dropout=0.25,
        fpn_channels=256,
        abn_block=ABN,
        activation=ACT_RELU,
        full_size_mask=True,
        deep_supervision=False
    ):
        super().__init__()
        self.encoder = encoder

        abn_block = partial(abn_block, activation=activation)

        self.decoder = FPNCatDecoder(
            encoder.channels,
            context_block=partial(FPNContextBlock, abn_block=abn_block),
            bottleneck_block=partial(FPNBottleneckBlock, abn_block=abn_block),
            fpn_channels=fpn_channels,
        )

        self.fuse = FPNFuse()
        self.segmentation_head = nn.Sequential(
            OrderedDict([("drop", nn.Dropout2d(dropout)), ("conv", conv1x1(sum(self.decoder.channels), num_classes))])
        )
        self.full_size_mask = full_size_mask
        self.deep_supervision = deep_supervision
        self.supervision = nn.ModuleList([conv1x1(channels, num_classes) for channels in self.decoder.channels])
    
    def forward(self, x):
        x_size = x.size()
        x = self.encoder(x)
        x = self.decoder(x)
        x_fuse = self.fuse(x)
        # Decode mask
        mask = self.segmentation_head(x_fuse)

        if self.full_size_mask:
            mask = F.interpolate(mask, size=x_size[2:], mode="bilinear", align_corners=False)

        prediction_list = []
        if self.deep_supervision:
            for feature_map, supervision  in zip(x, self.supervision):
                prediction = supervision(feature_map)
                prediction_list.append(prediction)
    
            return mask, prediction_list
        else:
            return mask

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



def resnet34_fpncat128(num_classes=5, dropout=0.0, pretrained=True, deep_supervision=False):
    encoder = E.Resnet34Encoder(pretrained=pretrained)
    return FPNCatSegmentationModel(encoder, num_classes=num_classes, fpn_channels=128, dropout=dropout, deep_supervision=deep_supervision)


def seresnext50_fpncat128(num_classes=5, dropout=0.0, pretrained=True, deep_supervision=False):
    encoder = E.SEResNeXt50Encoder(pretrained=pretrained)
    return FPNCatSegmentationModel(encoder, num_classes=num_classes, fpn_channels=128, dropout=dropout, deep_supervision=deep_supervision)


def seresnext101_fpncat256(num_classes=5, dropout=0.0, pretrained=True, deep_supervision=False):
    encoder = E.SEResNeXt101Encoder(pretrained=pretrained)
    return FPNCatSegmentationModel(encoder, num_classes=num_classes, fpn_channels=256, dropout=dropout, deep_supervision=deep_supervision)


def seresnext101_fpnsum256(num_classes=5, dropout=0.0, pretrained=True, deep_supervision=False):
    encoder = E.SEResNeXt101Encoder(pretrained=pretrained)
    return FPNSumSegmentationModel(encoder, num_classes=num_classes, fpn_channels=256, dropout=dropout, deep_supervision=deep_supervision)


def resnet152_fpncat256(num_classes=5, dropout=0.0, pretrained=True, deep_supervision=False):
    encoder = E.Resnet152Encoder(pretrained=pretrained)
    return FPNCatSegmentationModel(encoder, num_classes=num_classes, fpn_channels=256, dropout=dropout, deep_supervision=deep_supervision)


def effnetB4_fpncat128(num_classes=5, dropout=0.0, pretrained=True, deep_supervision=False):
    encoder = E.EfficientNetB4Encoder(activation= "swish")
    return FPNCatSegmentationModel(encoder, num_classes=num_classes, fpn_channels=128, dropout=dropout, deep_supervision=deep_supervision)

def b2_fpn_cat(input_channels=3, num_classes=1, dropout=0.2, pretrained=True, deep_supervision=False):
    encoder = B2Encoder(pretrained=pretrained, layers=[1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return FPNCatSegmentationModel(
        encoder, num_classes=num_classes, fpn_channels=64, activation=ACT_SWISH, dropout=dropout, deep_supervision=deep_supervision
    )


def b4_fpn_cat(input_channels=3, num_classes=1, dropout=0.2, pretrained=True):
    encoder = B4Encoder(pretrained=pretrained, layers=[1, 2, 3, 4])
    if input_channels != 3:
        encoder.change_input_channels(input_channels)

    return FPNCatSegmentationModel(
        encoder, num_classes=num_classes, fpn_channels=64, activation=ACT_SWISH, dropout=dropout
    )

if __name__ == '__main__':
    import torch
    a  = torch.randn(2, 3, 1024, 1024).cuda()
    model = resnet152_fpncat256(deep_supervision=True).cuda()
    # print(model.encoder)
    # print('-'*20)
    # print(model.decoder)
    output, deep_output = model(a)
    print(output.shape)
    for d in deep_output:
        print(d.shape)
