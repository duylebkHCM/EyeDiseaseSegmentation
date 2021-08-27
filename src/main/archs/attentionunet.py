"""
@author: Duy Le <leanhduy497@gmail.com>
"""
from pytorch_toolbelt.modules.dsconv import DepthwiseSeparableConv2d
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from pytorch_toolbelt.inference.functional import pad_image_tensor, unpad_image_tensor
from pytorch_toolbelt.modules.encoders.timm.efficient_net import make_n_channel_input_conv2d_same

import timm
from timm.models.efficientnet_blocks import DepthwiseSeparableConv, InvertedResidual
# import sys
# sys.path.append('..')
from .modules import *
from .model_util import init_weights, get_lr_parameters

__all__ = [
    'Attention_Unet', 
    'Unet_Encoder',
    'resnet50_attunet', 
    'efficientnetb2_attunet', 
    'mobilenetv3_attunet',
    'swin_tiny_attunet'
]

class Unet_Encoder(nn.Module):
    def __init__(self, freeze_bn=True, backbone: str ='resnext50d_32x4d', freeze_backbone=False, pretrained=True):
        super(Unet_Encoder, self).__init__()
        if pretrained:
            if backbone.startswith('swin'):
                from .modules.swin_transformer import create_model
                # pretrained_model = timm.create_model(backbone, pretrained=False, num_classes=1)
                # pretrained_model.load_state_dict(torch.load('models/IDRiD/pretrained_models/model.bin'))
                
                # pretrained_dict = pretrained_model.state_dict()
                # encoder_dict = self.encoder.state_dict()
                # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}

                # encoder_dict.update(pretrained_dict) 

                # self.encoder.load_state_dict(pretrained_dict)
                self.encoder = create_model(backbone)
                self.filters = self.encoder.num_features
            else:
                self.encoder = timm.create_model(backbone, features_only=True, pretrained=pretrained)
                self.filters = self.encoder.feature_info.channels()
            if freeze_bn:
                self.freeze_bn()
            if freeze_backbone:
                for param in self.encoder.parameters():
                    param.requires_grad = False
        else:
            if backbone.startswith('swin'):
                from .modules.swin_transformer import create_model
                self.encoder = create_model(backbone, pretrained=False)
                self.filters = self.encoder.num_features
            else:
                self.encoder = timm.create_model(backbone, features_only=True, pretrained=False)
                self.filters = self.encoder.feature_info.channels()
                init_weights(self)

    def change_input_channels(self, input_channels: int, mode="auto", **kwargs):
        self.encoder.conv_stem = make_n_channel_input_conv2d_same(
            self.encoder.conv_stem, input_channels, mode, **kwargs
        )
        return self

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def forward(self, input):
        encoder_features = self.encoder(input)
        return encoder_features


class EfficientUnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation=Swish, drop_block_rate=0.0):
        super().__init__()
        self.ir = InvertedResidual(in_channels, out_channels, act_layer=activation, se_ratio=0.25, exp_ratio=4)
        self.drop = DropBlock2D(drop_block_rate, 2)
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     activation(inplace=True),
        # )
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     activation(inplace=True),
        # )
        self.conv1 = nn.Sequential(
           DepthwiseSeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1) ,
           nn.BatchNorm2d(out_channels),
            activation(inplace=True),
        )
        self.conv2 = nn.Sequential(
           DepthwiseSeparableConv2d(out_channels, out_channels, kernel_size=3, padding=1) ,
           nn.BatchNorm2d(out_channels),
            activation(inplace=True),
        )

    def forward(self, x):
        x = self.ir(x)
        x = self.drop(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Up_Atten(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear = True):
        super(Up_Atten, self).__init__()
        self.atten = Attention_block(F_g=in_ch // 2, F_l=out_ch, F_int=in_ch)
        self.up_conv = DoubleConv(in_ch // 2 + out_ch, out_ch)
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, padding=1)
            )
        else:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=4, stride=2, padding=1)
        self.out_ch = out_ch

    def forward(self, input1, input2):
        d2 = self.up(input1) 
        d1 = self.atten(d2, input2)
        d2 = F.interpolate(d2, size=(d1.size(2), d1.size(3)), mode="bilinear", align_corners=True)
        d = torch.cat([d1, d2], dim=1)
        return self.up_conv(d)
    
class Unet_Decoder(nn.Module):
    def __init__(self, encoder_channels, n_classes, dropout):
        super(Unet_Decoder, self).__init__()
        self.decoder_output = nn.ModuleList()
        encoder_channels = encoder_channels[::-1]
        array_1 = encoder_channels[:-1]
        array_2 = encoder_channels[1:]

        for i, (in_ch, out_ch) in enumerate(zip(array_1, array_2)):
            next_up = Up_Atten(in_ch, out_ch) 
            self.decoder_output.append(next_up)
        self.dropout = nn.Dropout2d(dropout)
        self.out_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            OutConv(encoder_channels[-1], n_classes)
        )

        channels = [n_classes]
        channels += [feature.out_ch for feature in list(reversed(self.decoder_output))]
        self.channels = channels
        init_weights(self)
    
    def forward(self, encoder_features):    
        decoder_features = []  
        reverse_features = encoder_features[::-1]  
        up_decode = reverse_features[0]
        for i, feature in enumerate(reverse_features[1: ]):
            out_decode = self.decoder_output[i](up_decode, feature)
            decoder_features.append(out_decode)
            up_decode = out_decode
        final = self.dropout(up_decode)
        final = self.out_conv(final)
        decoder_features.append(final)
        return list(reversed(decoder_features))
        # return final

class Attention_Unet(nn.Module):
    """
        Attention Unet with pretrained model.
        Resnet18, resnet34, resnet50, resnet101, wide_resnet, ... from timm package
    """
    def __init__(self, n_classes, dropout, deep_supervision:bool, encoder: nn.Module):
        super(Attention_Unet, self).__init__()
        
        self.encoder = encoder
        encoder_channels = self.encoder.filters
        self.decoder = Unet_Decoder(encoder_channels, n_classes, dropout)
        self.deep_supervision = deep_supervision
        # if deep_supervision:
        self.supervision = nn.ModuleList([OutConv(channels, n_classes) for channels in self.decoder.channels])

    def forward(self, x):
        x, pad = pad_image_tensor(x, 32)
        H, W = x.size(2), x.size(3)

        #Encode
        encoder_outputs = self.encoder(x)
        #Decode
        decoder_outputs = self.decoder(encoder_outputs)
        # if the input is not divisible by the output stride
        final = decoder_outputs[0]
        if final.size(2) != H or final.size(3) != W:
            final = F.interpolate(final, size=(H, W), mode="bilinear", align_corners=True)
        
        final = unpad_image_tensor(final, pad)
        prediction_list = []
        if self.deep_supervision:
            for feature_map, supervision  in zip(decoder_outputs, self.supervision):
                prediction = supervision(feature_map)
                prediction_list.append(prediction)
    
            return final, prediction_list[1: ]
        else:
            return final

    def get_num_parameters(self):
        trainable= int(sum(p.numel() for p in self.parameters() if p.requires_grad))
        total = int(sum(p.numel() for p in self.parameters()))
        return trainable, total
    
    def get_paramgroup(self, base_lr=None, weight_decay=1e-5):
        lr_dict = {
            "encoder": [0.1, weight_decay],
        }
        
        lr_group = get_lr_parameters(self, base_lr, lr_dict)
        return lr_group

def seresnet50_attunet(num_classes=1, drop_rate=0.25, pretrained=True, freeze_bn=True, freeze_backbone=False, deep_supervision=False):
    encoder = Unet_Encoder(freeze_bn=freeze_bn, backbone='seresnet50', freeze_backbone=freeze_backbone, pretrained=pretrained)
    return Attention_Unet(n_classes=num_classes,dropout=drop_rate, encoder=encoder, deep_supervision=deep_supervision)

def seresnet50_attunet(num_classes=1, drop_rate=0.25, pretrained=True, freeze_bn=True, freeze_backbone=False, deep_supervision=False):
    encoder = Unet_Encoder(freeze_bn=freeze_bn, backbone='seresnet50', freeze_backbone=freeze_backbone, pretrained=pretrained)
    return Attention_Unet(n_classes=num_classes,dropout=drop_rate, encoder=encoder, deep_supervision=deep_supervision)

def resnet50_attunet(num_classes=1, drop_rate=0.25, pretrained=True, freeze_bn=True, freeze_backbone=False, deep_supervision=False):
    encoder = Unet_Encoder(freeze_bn=freeze_bn, backbone='resnet50', freeze_backbone=freeze_backbone, pretrained=pretrained)
    return Attention_Unet(n_classes=num_classes,dropout=drop_rate, encoder=encoder, deep_supervision=deep_supervision)

def efficientnetb2_attunet(input_channels = 3, num_classes=1, drop_rate=0.25, pretrained=True, freeze_bn=True, freeze_backbone=False,  deep_supervision=False):
    encoder = Unet_Encoder(freeze_bn=freeze_bn, backbone='tf_efficientnet_b2', freeze_backbone=freeze_backbone, pretrained=pretrained)
    if input_channels != 3:
        encoder.change_input_channels(input_channels)
    return Attention_Unet(n_classes=num_classes,dropout=drop_rate, encoder=encoder, deep_supervision=deep_supervision)

def mobilenetv3_attunet(num_classes=1, drop_rate=0.25, pretrained=True, freeze_bn=True, freeze_backbone=False):
    encoder = Unet_Encoder(freeze_bn=freeze_bn, backbone='mobilenetv3_large_100', freeze_backbone=freeze_backbone, pretrained=pretrained)
    return Attention_Unet(n_classes=num_classes,dropout=drop_rate, encoder=encoder)

def swin_tiny_attunet(num_classes=1, drop_rate=0.25, drop_block_rate=0.1, pretrained=True, freeze_bn=True, freeze_backbone=False, deep_supervision=False):
    encoder = Unet_Encoder(backbone='swin_tiny_patches4_window7_224', freeze_bn=freeze_bn, freeze_backbone=freeze_backbone, pretrained=pretrained)
    return Attention_Unet(n_classes=num_classes,dropout=drop_rate, drop_block_rate=drop_block_rate, deep_supervision=deep_supervision, encoder=encoder)

def swin_small_attunet(num_classes=1, drop_rate=0.25, drop_block_rate=0.1, pretrained=True, freeze_bn=True, freeze_backbone=False, deep_supervision=False):
    encoder = Unet_Encoder(backbone='swin_small_patches4_window7_224', freeze_bn=freeze_bn, freeze_backbone=freeze_backbone, pretrained=pretrained)
    return Attention_Unet(n_classes=num_classes,dropout=drop_rate, drop_block_rate=drop_block_rate, deep_supervision=deep_supervision, encoder=encoder)

if __name__ == '__main__':
    model = resnet50_attunet(1, deep_supervision=True).cuda()
    a = torch.randn(1, 3, 1024, 1024).cuda()
    output, deep_output = model(a)
    # for o in output:
    #     print(o.shape
    # 
    # )
    print(output.shape)
    for fea in deep_output:
        print(fea.shape)
    # decoder = model.decoder
    # decoder_fea = decoder.channels
    # for chn in decoder_fea:
    #     print(chn)
