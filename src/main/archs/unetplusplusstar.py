# from src.main.archs.modules.vit_res_encoder import conv3x3
from pytorch_toolbelt.modules.backbone.senet import se_resnet50
import torch
from torch import nn
from torch.nn import functional as F
from segmentation_models_pytorch.base import modules as md
from typing import Optional, Union, List
from timm.models.layers import DropPath, DropBlock2d
# import sys
# sys.path.append('.')
# from .modules import DropBlock2D
from einops import rearrange
from .modules import BottleBlock
from .axial_attention_v2 import AxialAttentionBlock, CrossAxialAttention, AxialAttention
from .model_util import add_weight_decay, get_lr_parameters
try:
    from inplace_abn import InPlaceABN
except:
    print("In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn")
    
class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
            drop_block_prob = 0.1
    ):

        if use_batchnorm == "inplace" and InPlaceABN is None:
            raise RuntimeError(
                "In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn"
            )

        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not(use_batchnorm),
        )

        dropblock = DropBlock2d(block_size=7, drop_prob=drop_block_prob)

        relu = nn.ReLU(inplace=True)

        if use_batchnorm == "inplace":
            bn = InPlaceABN(out_channels, activation="leaky_relu", activation_param=0.0)
            relu = nn.Identity()

        elif use_batchnorm and use_batchnorm != "inplace":
            bn = nn.BatchNorm2d(out_channels)

        else:
            bn = nn.Identity()

        super(Conv2dReLU, self).__init__(conv, dropblock, bn, relu)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            base_dim=32,
            level=0,
            use_catt=False,
            use_batchnorm=True,
            attention_type=None,
            drop_block_prob=0.1
    ):
        super().__init__()
        self.dim=base_dim*(2**level)
        self.use_catt = use_catt
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
            drop_block_prob=drop_block_prob
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
            drop_block_prob=drop_block_prob
        )

        if use_catt:
            self.init_conv = nn.Sequential(
                nn.MaxPool2d(2),
                nn.Conv2d(skip_channels, skip_channels//16, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(skip_channels//16),
                nn.ReLU(True)
            )

            self.h_catt = CrossAxialAttention(self.dim, in_channels, skip_channels//16, heads=4, dim_head_kq=8)
            self.w_catt = CrossAxialAttention(self.dim, in_channels, skip_channels//16, heads=4, dim_head_kq=8)

            self.up_scale = nn.Sequential(
                nn.Sigmoid(),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)   
            )
            self.down_sample = conv1x1(skip_channels, skip_channels//16)
            self.up_sample = conv1x1(skip_channels//16, skip_channels)
        else:
            if skip_channels > 0:
                self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
            self.attention2 = md.Attention(attention_type, in_channels=out_channels)


    def forward(self, x, skip=None):
        x_up = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip is not None:
            if self.use_catt:
                ori_skip = self.down_sample(skip)

                skip = self.init_conv(skip)

                x_1 = rearrange(x, 'b c h w -> (b w) c h')
                skip = rearrange(skip, 'b c h w -> (b w) c h')

                skip = self.h_catt(x_1, skip)

                x_2 = rearrange(x, 'b c h w -> (b h) c w')
                skip = rearrange(skip, '(b w) c h  -> (b h) c w', w=self.dim)

                skip = self.w_catt(x_2, skip)
                skip = rearrange(skip, '(b h) c w -> b c h w', h=self.dim)
                
                skip = self.up_scale(skip)
                skip = ori_skip*skip
                skip = self.up_sample(skip)

                x_up = torch.cat([x_up, skip], dim=1)
            else:

                x = torch.cat([x_up, skip], dim=1)
                x_up = self.attention1(x)

        x = self.conv1(x_up)
        x = self.conv2(x)
        if not self.use_catt:
            x = self.attention2(x)

        return x

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = md.Activation(activation)
        super().__init__(conv2d, upsampling, activation)

class ClassificationHead(nn.Sequential):
    def __init__(self, in_channels, classes, pooling="avg", dropout=0.2, activation=None):
        if pooling not in ("max", "avg"):
            raise ValueError("Pooling should be one of ('max', 'avg'), got {}.".format(pooling))
        pool = nn.AdaptiveAvgPool2d(1) if pooling == 'avg' else nn.AdaptiveMaxPool2d(1)
        flatten = md.Flatten()
        dropout = nn.Dropout(p=dropout, inplace=True) if dropout else nn.Identity()
        linear = nn.Linear(in_channels, classes, bias=True)
        activation = md.Activation(activation)
        super().__init__(pool, flatten, dropout, linear, activation)

class UnetPlusPlusDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            base_dim=32,
            use_batchnorm=True,
            attention_type=None,
            deep_supervision=False,
            drop_block_prob=0.1,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        self.in_channels = [head_channels] + list(decoder_channels[:-1])
        self.skip_channels = list(encoder_channels[1:]) + [0]
        self.out_channels = decoder_channels

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type, drop_block_prob=drop_block_prob, base_dim=base_dim)

        blocks = {}
        for layer_idx in range(len(self.in_channels) - 1): 
            for depth_idx in range(layer_idx+1):
                if depth_idx == 0:
                    in_ch = self.in_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx+1)
                    out_ch = self.out_channels[layer_idx]
                else:
                    out_ch = self.skip_channels[layer_idx]
                    skip_ch = self.skip_channels[layer_idx] * (layer_idx+1-depth_idx)
                    in_ch = self.skip_channels[layer_idx - 1]

                if layer_idx ==0 or layer_idx ==1:
                    use_catt=True
                else:
                    use_catt=False

                blocks[f'x_{depth_idx}_{layer_idx}'] = DecoderBlock(in_ch, skip_ch, out_ch, use_catt=use_catt, level=layer_idx, **kwargs)

        blocks[f'x_{0}_{len(self.in_channels)-1}'] = DecoderBlock(self.in_channels[-1], 0, self.out_channels[-1], use_catt=False, **kwargs)
       
        self.blocks = nn.ModuleDict(blocks)
        self.depth = len(self.in_channels) - 1
        self.deep_supervision = deep_supervision

    def forward(self, features: List[torch.Tensor]):
        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder
        
        # start building dense connections
        dense_x = {}
        for layer_idx in range(len(self.in_channels)-1):
            for depth_idx in range(self.depth-layer_idx):
                if layer_idx == 0:
                    output = self.blocks[f'x_{depth_idx}_{depth_idx}'](features[depth_idx], features[depth_idx+1])
                    dense_x[f'x_{depth_idx}_{depth_idx}'] = output

                else:
                    dense_l_i = depth_idx + layer_idx
                    cat_features = [dense_x[f'x_{idx}_{dense_l_i}'] for idx in range(depth_idx+1, dense_l_i+1)]
                    cat_features = torch.cat(cat_features + [features[dense_l_i+1]], dim=1)
                    dense_x[f'x_{depth_idx}_{dense_l_i}'] =\
                        self.blocks[f'x_{depth_idx}_{dense_l_i}'](dense_x[f'x_{depth_idx}_{dense_l_i-1}'], cat_features)

        dense_x[f'x_{0}_{self.depth}'] = self.blocks[f'x_{0}_{self.depth}'](dense_x[f'x_{0}_{self.depth-1}'])
        
        if self.deep_supervision:
            return dense_x[f'x_{0}_{self.depth}'], [dense_x['x_3_3'], dense_x['x_2_3'], dense_x['x_1_3']]
        else:
            return dense_x[f'x_{0}_{self.depth}']
    
def pair(x):
    return (x, x) if not isinstance(x, tuple) else x

def set_bn_eval(module: nn.Module):
    "Set bn layers in eval mode for all recursive children of `m`."
    bn_types = nn.BatchNorm2d

    if isinstance(module, bn_types):
        module.track_running_stats = False

    for m in module.modules():
        if isinstance(m, bn_types):
            module.track_running_stats = False

class BoTSER50(nn.Module):
    def __init__(self, base_dim=32, pretrained=True, use_axial=False, num_transblocks=1):
        super().__init__()
        seresnet = se_resnet50(pretrained=  None)
        if pretrained:
            seresnet.load_state_dict(torch.load('models/pretrained_models/se_resnet50-ce0d4300.pth', map_location='cpu'))

        self.maxpool = seresnet.layer0.pool
        del seresnet.layer0.pool

        self.layerinit = nn.Identity()
        self.layer0 = seresnet.layer0
        self.layer1 = seresnet.layer1
        self.layer2 = seresnet.layer2
        self.layer3 = seresnet.layer3

        if use_axial:
            block = AxialAttentionBlock(
                in_channels=2048, 
                dim=base_dim,
                out_channels=2048,
                down_sample=False,
                heads=8
            )
            first_block = AxialAttentionBlock(
                in_channels=1024, 
                dim=base_dim*2,
                out_channels=2048,
                down_sample=True,
                heads=8
            )
        else:
            block = BottleBlock(
                dim=2048,
                fmap_size=pair(32),
                dim_out=2048,
                proj_factor=4,
                downsample=False,
                heads=8,
                dim_head=128,
                rel_pos_emb=True,
                activation=nn.ReLU()
            )

        self.layer4 = nn.Sequential(
            # *list(seresnet.layer4[:1]),
            first_block,
            block,
            block
        )

        out_channels = [3, 64, 256, 512, 1024, 2048]
        self.out_channels = out_channels

        if pretrained:
            for layer  in [self.layer0, self.layer1, self.layer2, self.layer3]:
                set_bn_eval(layer)
    
    @property
    def encoder_layers(self):
        return [self.layerinit, self.layer0, self.layer1, self.layer2, self.layer3, self.layer4]

    def forward(self, x):
        output_features = []
        for layer in self.encoder_layers:
            output = layer(x)
            output_features.append(output)

            if layer == self.layer0:
                output = self.maxpool(output)

            x = output

        return output_features

def initialize(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode="fan_in", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

encoder_config = {
    'BoTSER50_Axial_Imagenet': {
        'pretrained': True,
        'use_axial': True,
        'num_transblocks': 1
    },
    'BoTSER50_Axial_Imagenet_2': {
    'pretrained': True,
    'use_axial': True,
    'num_transblocks':2
    },
    'BoTSER50_Axial_Imagenet_3': {
    'pretrained': True,
    'use_axial': True,
    'num_transblocks': 3,
    },
    'BoTSER50_Axial_scratch': {
        'pretrained': False,
        'use_axial': True
    },
    'BoTSER50_Imagenet': {
        'pretrained': True,
        'use_axial': False
    }
}

def get_encoder(base_dim, encoder_name:str):
    kwargs = encoder_config[encoder_name]
    return BoTSER50(base_dim=base_dim, **kwargs)

class UnetPlusPlusStar(nn.Module):
    def __init__(
            self,
            encoder_name,
            encoder_depth: int = 5,
            decoder_use_batchnorm: bool = True,
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            decoder_attention_type: Optional[str] = None,
            base_dim=32, 
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,
            deep_supervision=False,
            drop_block_prob=0.1,
            clf_head=False
        ):
            super().__init__()
            self.encoder = get_encoder(base_dim,encoder_name)

            self.decoder = UnetPlusPlusDecoder(
                encoder_channels=self.encoder.out_channels,
                decoder_channels=decoder_channels,
                n_blocks=encoder_depth,
                base_dim=base_dim,
                use_batchnorm=decoder_use_batchnorm,
                attention_type=decoder_attention_type,
                deep_supervision=deep_supervision,
                drop_block_prob=drop_block_prob,
            )

            self.segmentation_head = SegmentationHead(
                in_channels=decoder_channels[-1],
                out_channels=classes,
                activation=activation,
                kernel_size=3,
            )
            
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1],
                classes=classes,
                dropout=0.1
            )

            self.deep_segmentation_head = nn.ModuleList(
                [
                    SegmentationHead(
                        in_channels=decoder_channels[-3],
                        out_channels=classes,
                        activation=activation,
                        kernel_size=3
                    ) for _ in range(3)
                ]
            )

            self.clf_head = clf_head
            self.deep_supervision = deep_supervision
            self.name = f"unetplusplus-{encoder_name}"
            self.initialize()

    def initialize(self):
        initialize(self.decoder)
        initialize(self.segmentation_head)
        initialize(self.deep_segmentation_head)
        if self.clf_head:
            initialize(self.classification_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        clf = self.classification_head(features[-1])
        decoder_output = self.decoder(features)

        if self.deep_supervision:
            final_output = decoder_output[0]
            deep_outputs = decoder_output[1]
            final_mask = self.segmentation_head(final_output)
            masks = []
            for feature, segmentation_head in zip(deep_outputs, self.deep_segmentation_head):
                masks.append(segmentation_head(feature))

            if self.clf_head:
                return final_mask, masks, clf
            return final_mask, masks
        else:
            final_output = decoder_output
            mask = self.segmentation_head(decoder_output)

            if self.clf_head:
                return mask, clf
            return mask

    def get_num_parameters(self):
        trainable= int(sum(p.numel() for p in self.parameters() if p.requires_grad))
        total = int(sum(p.numel() for p in self.parameters()))
        return trainable, total
    
    
    def get_paramgroup(self, base_lr=None, weight_decay=1e-5):
        lr_dict = {
            "encoder.layer0": [0.1, weight_decay],
            "encoder.layer1":[0.1, weight_decay],
            "encoder.layer2": [0.1, weight_decay],
            "encoder.layer3": [0.1, weight_decay],
            "encoder.layer4.0.height_att.RelativePosEncQKV": [1.0, 0.0],
            "encoder.layer4.0.width_att.RelativePosEncQKV": [1.0, 0.0],
            "encoder.layer4.1.height_att.RelativePosEncQKV": [1.0, 0.0],
            "encoder.layer4.1.width_att.RelativePosEncQKV": [1.0, 0.0],
            "encoder.layer4.2.height_att.RelativePosEncQKV": [1.0, 0.0],
            "encoder.layer4.2.width_att.RelativePosEncQKV": [1.0, 0.0],
            "decoder.blocks.x_0_0.h_catt.RelativePosEncQKV": [1.0, 0.0],
            "decoder.blocks.x_0_0.w_catt.RelativePosEncQKV": [1.0, 0.0],
            "decoder.blocks.x_0_1.h_catt.RelativePosEncQKV": [1.0, 0.0],
            "decoder.blocks.x_0_1.w_catt.RelativePosEncQKV": [1.0, 0.0],
            "decoder.blocks.x_1_1.h_catt.RelativePosEncQKV": [1.0, 0.0],
            "decoder.blocks.x_1_1.w_catt.RelativePosEncQKV": [1.0, 0.0]
        }
        
        lr_group = get_lr_parameters(self, base_lr, lr_dict)
        return lr_group


if __name__ == '__main__':
    import torch.cuda.amp as amp
    model = UnetPlusPlusStar(
        encoder_name = 'BoTSER50_Axial_Imagenet_3', 
        decoder_attention_type='scse', 
        decoder_use_batchnorm=True, 
        deep_supervision=False, 
        base_dim=32, 
        drop_block_prob=0.0)
        
    # for pred in preds:
    #     print(pred.shape)


    # optim = torch.optim.Adam(param_group, lr = 0.001, weight_decay=1e-5)
    # for param_group in optim.param_groups:
    #     print(float(param_group['lr']))

    model.load_state_dict(torch.load('../../../models/IDRiD/EX/Jul16_12_27/checkpoints/best.pth', map_location='cpu')['model_state_dict'])
    model=model.cuda()
    print('Number params', model.get_num_parameters())
    img = torch.randn(2, 3, 1024, 1024).cuda()
    with amp.autocast():
        final_pred = model(img) 

    print(final_pred.shape)

    loss = nn.BCEWithLogitsLoss()(final_pred, torch.zeros(2, 1, 1024, 1024).cuda())

    loss.backward()


