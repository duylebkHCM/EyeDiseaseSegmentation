# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# import sys
# sys.path.append('..')

from ..util.checkpoint import load_checkpointv2
from .model_util import add_weight_decay
import math
from .modules.swin_transformer import create_model

# BACKBONE = {
#     'mit_b0': {
#         'pretrained': 'models/pretrained_models/mit_b0.pth',
#         'model': mit_b0()
#     },
#     'mit_b1': {
#         'pretrained': 'models/pretrained_models/mit_b1.pth',
#         'model': mit_b1()
#     },
#     'mit_b2': {
#         'pretrained': 'models/pretrained_models/mit_b2.pth',
#         'model': mit_b2()
#     },
# }

def init_weight(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()


def conv3x3(in_channel, out_channel): #not change resolusion
    return nn.Conv2d(in_channel,out_channel,
                      kernel_size=3,stride=1,padding=1,dilation=1,bias=False)

def conv1x1(in_channel, out_channel): #not change resolution
    return nn.Conv2d(in_channel,out_channel,
                      kernel_size=1,stride=1,padding=0,dilation=1,bias=False)


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channel, reduction):
        super().__init__()
        self.global_maxpool = nn.AdaptiveMaxPool2d(1)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1) 
        self.fc = nn.Sequential(
            conv1x1(in_channel, in_channel//reduction).apply(init_weight),
            nn.ReLU(True),
            conv1x1(in_channel//reduction, in_channel).apply(init_weight)
        )
        
    def forward(self, inputs):
        x1 = self.global_maxpool(inputs)
        x2 = self.global_avgpool(inputs)
        x1 = self.fc(x1)
        x2 = self.fc(x2)
        x  = torch.sigmoid(x1 + x2)
        return x

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3x3 = conv3x3(2,1).apply(init_weight)
        
    def forward(self, inputs):
        x1,_ = torch.max(inputs, dim=1, keepdim=True)
        x2 = torch.mean(inputs, dim=1, keepdim=True)
        x  = torch.cat([x1,x2], dim=1)
        x  = self.conv3x3(x)
        x  = torch.sigmoid(x)
        return x
    
class CBAM(nn.Module):
    def __init__(self, in_channel, reduction):
        super().__init__()
        self.channel_attention = ChannelAttentionModule(in_channel, reduction)
        self.spatial_attention = SpatialAttentionModule()
        
    def forward(self, inputs):
        x = inputs * self.channel_attention(inputs)
        x = x * self.spatial_attention(x)
        return x
    
    
class CenterBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = conv3x3(in_channel, out_channel).apply(init_weight)
        self.In1 = nn.InstanceNorm2d(in_channel).apply(init_weight)

    def forward(self, inputs):
        x = self.conv(inputs)
        x = F.relu(self.In1(x))
        return x

class DecodeBlock(nn.Module):
    def __init__(self, in_channel, out_channel, upsample):
        super().__init__()
        self.In1 = nn.InstanceNorm2d(in_channel).apply(init_weight)
        self.upsample = nn.Sequential()
        if upsample:
            self.upsample.add_module('upsample',nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.conv3x3_1 = conv3x3(in_channel, in_channel).apply(init_weight)
        self.In2 = nn.InstanceNorm2d(in_channel).apply(init_weight)
        self.conv3x3_2 = conv3x3(in_channel, out_channel).apply(init_weight)
        self.cbam = CBAM(out_channel, reduction=16)
        self.conv1x1 = conv1x1(in_channel, out_channel).apply(init_weight)
        
    def forward(self, inputs):
        x  = inputs
        x  = self.upsample(x)
        x  = self.conv3x3_1(x)
        x  = F.relu(self.In1(self.conv3x3_2(x)))
        x  = self.cbam(x)
        x += F.relu(self.In2(self.conv1x1(self.upsample(inputs)))) #shortcut
        return x

class SwinformerStar(nn.Module):
    def __init__(self, backbone, deep_supervision, clfhead, pretrained=True):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.clfhead = clfhead

        swin_transformer = create_model(backbone, pretrained=pretrained)
        self.encoder = swin_transformer
        encoder_features = swin_transformer.num_features

        self.center_block = CenterBlock(encoder_features[-1], encoder_features[-1])

        self.decoder4 = DecodeBlock(encoder_features[-1] + encoder_features[-1], 64, upsample=True)
        self.decoder3 = DecodeBlock(encoder_features[-2] + 64, 64, upsample=True)
        self.decoder2 = DecodeBlock(encoder_features[-3] + 64, 64, upsample=True)
        self.decoder1 = DecodeBlock(encoder_features[-4] + 64, 64, upsample=True)
        self.decoder0 = DecodeBlock(64, 64, upsample=True)

        #upsample
        self.upsample4 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)
        self.upsample3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)
        self.upsample2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        #deep supervision
        self.deep4 = conv1x1(64,1).apply(init_weight)
        self.deep3 = conv1x1(64,1).apply(init_weight)
        self.deep2 = conv1x1(64,1).apply(init_weight)
        self.deep1 = conv1x1(64,1).apply(init_weight)
        #final conv
        self.final_conv = conv1x1(64,1).apply(init_weight)

        #Queries
        self.que4 = conv1x1(64,1).apply(init_weight)
        self.que3 = conv1x1(64,1).apply(init_weight)
        self.que2 = conv1x1(64,1).apply(init_weight)
        self.que1 = conv1x1(64,1).apply(init_weight)
        self.que0 = conv1x1(64,1).apply(init_weight)

        #clf head
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.clf = nn.Sequential(
            nn.LayerNorm(encoder_features[-1]).apply(init_weight),
            nn.Linear(encoder_features[-1],256).apply(init_weight),
            nn.ELU(True),
            nn.LayerNorm(256).apply(init_weight),
            nn.Linear(256,1).apply(init_weight)
        )

    def forward(self, x):
        B, _, H, W = x.shape
        x1, x2, x3, x4 = self.encoder(x)
        
        #clf head
        logits_clf = self.clf(self.avgpool(x4).squeeze(-1).squeeze(-1)) #->(*,1)

        y5 = self.center_block(x4)
        y4 = self.decoder4(torch.cat([x4, y5], dim=1))
        y3 = self.decoder3(torch.cat([x3, y4], dim=1))
        y2 = self.decoder2(torch.cat([x2, y3], dim=1))
        y1 = self.decoder1(torch.cat([x1, y2], dim=1))
        y0 = self.decoder0(y1)

        #hypercolumns
        y4 = self.upsample4(y4) #->(*,64,h,w)
        y3 = self.upsample3(y3) #->(*,64,h,w)
        y2 = self.upsample2(y2) #->(*,64,h,w)
        y1 = self.upsample1(y1) #->(*,64,h,w)
        s4 = self.deep4(y4)
        s3 = self.deep3(y3)
        s2 = self.deep2(y2)
        s1 = self.deep1(y1)
        s0 = self.final_conv(y0)
        predictions = torch.cat([s0, s1, s2, s3, s4], dim=1)

        q4 = self.que4(y4)
        q3 = self.que3(y3)
        q2 = self.que2(y2)
        q1 = self.que1(y1)
        q0 = self.que0(y0)
        queries = torch.cat([q0, q1, q2, q3, q4], dim=1)

        queries = queries.reshape(B, -1, 1, H, W)
        attn = F.softmax(queries, dim=1)
        predictions = predictions.reshape(B, -1, 1, H, W)
        combined_prediction = torch.sum(attn * predictions, dim=1)

        if self.clfhead:
            if self.deep_supervision:
                logits_deeps = [s4,s3,s2,s1]
                return combined_prediction, logits_deeps, logits_clf
            else:
                return combined_prediction, logits_clf
        else:
            if self.deep_supervision:
                logits_deeps = [s4,s3,s2,s1]
                return combined_prediction, logits_deeps
            else:
                return combined_prediction

    def get_num_parameters(self):
        trainable= int(sum(p.numel() for p in self.parameters() if p.requires_grad))
        total = int(sum(p.numel() for p in self.parameters()))
        return trainable, total
    
    def get_paramgroup(self, weight_decay=1e-5):
        lr_group = add_weight_decay(self, weight_decay=weight_decay, skip_list=())
        return lr_group

if __name__ == '__main__':
    import torch.cuda.amp as amp

    model = SwinformerStar('mit_b0', True, True).cuda()
    print('Model structure')
    for name, param in model.named_parameters():
        print('Name', name, 'Param shape', param.shape)

    print('Number of parameters', model.get_num_parameters())
    a = torch.rand(2, 3, 1024, 1024).cuda()

    with amp.autocast():
        final, d_final, clf = model(a)

    print('Final', final.shape)
    for o in d_final:
        print(o.shape)
    
    print('Clf', clf.shape)