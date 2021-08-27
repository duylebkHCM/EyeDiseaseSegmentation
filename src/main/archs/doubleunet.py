#Convert from TF source code from https://github.com/DebeshJha/2020-CBMS-DoubleU-Net/blob/master/model.py
"""
@author: Duy Le <leanhduy497@gmail.com>
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import logging
logging.basicConfig(level=logging.INFO)
from .modules import *   
from .model_util import init_weights

__all__ = [
    'Double_Unet',
    'Encoder1',
    'resnet50_doubleunet',
    'efficientnetb2_doubleunet',
    'mobilenetv3_doubleunet',
]

class Custom_Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Custom_Up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels + out_channels, out_channels, in_channels // 2)
            self.conv1 = DoubleConv(in_channels + 2*out_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels + out_channels, out_channels)
            self.conv1 = DoubleConv(in_channels + 2*out_channels, out_channels)

        self.se = SE_Block(out_channels, r=8)

    def forward(self, x1, x2, x3=None):
        x1 = self.up(x1)
        x1 = F.interpolate(x1, size=(x2.size(2), x2.size(3)), mode="bilinear", align_corners=True)
        if x3 is not None:
            x = torch.cat([x3, x2, x1], dim=1)
            x = self.conv1(x)
        else:
            x = torch.cat([x2, x1], dim=1)
            x = self.conv(x)
        x = self.se(x)
        return x

class Encoder1(nn.Module):
    def __init__(self, freeze_bn=True, backbone='resnext50d_32x4d', freeze_backbone=False, pretrained=True):
        super(Encoder1, self).__init__()
        if backbone:
            self.encoder = timm.create_model(backbone, features_only=True, pretrained=pretrained)
            self.filters = self.encoder.feature_info.channels()
            if freeze_bn:
                self.freeze_bn()
            if freeze_backbone:
                for param in self.encoder.parameters():
                    param.requires_grad = False

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def forward(self, input):
        encoder_features = self.encoder(input)
        return encoder_features


class Decoder1(nn.Module):
    def __init__(self, n_classes, encoder_channels):
        super(Decoder1, self).__init__()
        self.encoder_channels = encoder_channels[::-1]
        self.decoder_output = nn.ModuleList()
        array_1 = self.encoder_channels[:-1]
        array_2 = self.encoder_channels[1:]

        for i, (in_ch, out_ch) in enumerate(zip(array_1, array_2)):
            next_up = Custom_Up(in_ch, out_ch)
            self.decoder_output.append(next_up)
        self.clf = nn.Sequential(
            nn.ConvTranspose2d(self.encoder_channels[-1], self.encoder_channels[-1], 4, 2, 1),
            OutConv(self.encoder_channels[-1], n_classes)
        )
        init_weights(self)

    def forward(self, encoder_features):      
        reverse_features = encoder_features[::-1]  
        up_decode = reverse_features[0]
        for i, feature in enumerate(reverse_features[1: ]):
            out_decode = self.decoder_output[i](up_decode, feature)
            up_decode = out_decode
        final = self.clf(up_decode)
        return final


class Encoder2(nn.Module):
    def __init__(self, encoder1_channels):
        super(Encoder2, self).__init__()
        self.encoder1_channels = encoder1_channels  
        self.blocks =nn.ModuleList()
        array_1 = self.encoder1_channels[:-1]
        array_2 = self.encoder1_channels[1:]
        self.blocks.append(Down(3, array_1[0]))
        for i, (f_in, f_out) in enumerate(zip(array_1, array_2)):
            self.blocks.append(Down(f_in, f_out))
        init_weights(self)

    def forward(self, inputs):
        x = inputs
        skip_connections = []
        for block in self.blocks:
            x = block(x)
            skip_connections.append(x)
        return skip_connections


class Decoder2(nn.Module):
    def __init__(self, n_classes, dropout, encoder1_channels, encoder2_channels):
        super(Decoder2, self).__init__()
        self.encoder1_channels = encoder1_channels[::-1]
        self.encoder2_channels = encoder2_channels[::-1]
        self.decoder_output = nn.ModuleList()
        array_1 = self.encoder1_channels[:-1]
        array_2 = self.encoder1_channels[1:]

        for i, (in_ch, out_ch) in enumerate(zip(array_1, array_2)):
            next_up = Custom_Up(in_ch, out_ch)
            self.decoder_output.append(next_up)
        self.clf = nn.Sequential(
            nn.ConvTranspose2d(self.encoder2_channels[-1], self.encoder2_channels[-1], 4, 2, 1),
            OutConv(self.encoder2_channels[-1], n_classes)
        )
        self.dropout = nn.Dropout2d(dropout)
        init_weights(self)

    def forward(self, encoder1_features, encoder2_features):      
        reverse_features_1 = encoder1_features[::-1]  
        reverse_features_2 = encoder2_features[::-1]
        up_decode = reverse_features_2[0]
        for i, (feature1, feature2) in enumerate(zip(reverse_features_1[1: ], reverse_features_2[1: ])):
            out_decode = self.decoder_output[i](up_decode, feature1, feature2)
            up_decode = out_decode
        final = self.dropout(up_decode)
        final = self.clf(final)
        return final


class Double_Unet(nn.Module):
    def __init__(self, n_classes, dropout, encoder: nn.Module):
        super(Double_Unet, self).__init__()
        self.encoder = encoder
        encoder1_channels = self.encoder.filters
        self.decoder1 = Decoder1(n_classes, encoder1_channels)
        self.encoder2 = Encoder2(encoder1_channels)
        self.decoder2 = Decoder2(n_classes, dropout, encoder1_channels, encoder1_channels)
        self.aspp1 = ASPP(encoder1_channels[-1], 16, encoder1_channels[-1])
        self.aspp2 = ASPP(encoder1_channels[-1], 16, encoder1_channels[-1])
        
    def forward(self, input):
        encoder1_f = self.encoder(input)
        x = self.aspp1(encoder1_f[-1])       
        encoder1_f[-1] = x
        output1 = self.decoder1(encoder1_f)

        se_inputs =input*output1

        encoder2_f = self.encoder2(se_inputs)
        x = self.aspp2(encoder2_f[-1])
        encoder2_f[-1] = x
        output2 = self.decoder2(encoder1_f, encoder2_f)
        cat_output = torch.cat([output1*0.2, output2*0.8], dim=1)

        return torch.sum(cat_output, dim=1, keepdim=True)

def resnet50_doubleunet(num_classes=1, drop_rate=0.25, pretrained=True, freeze_bn=True, freeze_backbone=False):
    encoder = Encoder1(freeze_bn=freeze_bn, backbone='resnet50', freeze_backbone=freeze_backbone, pretrained=pretrained)
    return Double_Unet(n_classes=num_classes,dropout=drop_rate, encoder=encoder)

def efficientnetb2_doubleunet(num_classes=1, drop_rate=0.25, pretrained=True, freeze_bn=True, freeze_backbone=False):
    encoder = Encoder1(freeze_bn=freeze_bn, backbone='tf_efficientnet_b2', freeze_backbone=freeze_backbone, pretrained=pretrained)
    return Double_Unet(n_classes=num_classes,dropout=drop_rate, encoder=encoder)

def mobilenetv3_doubleunet(num_classes=1, drop_rate=0.25, pretrained=True, freeze_bn=True, freeze_backbone=False):
    encoder = Encoder1(freeze_bn=freeze_bn, backbone='mobilenetv3_large_100', freeze_backbone=freeze_backbone, pretrained=pretrained)
    return Double_Unet(n_classes=num_classes,dropout=drop_rate, encoder=encoder)
