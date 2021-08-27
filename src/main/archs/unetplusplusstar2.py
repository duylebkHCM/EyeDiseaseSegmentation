from pytorch_toolbelt.modules.backbone.senet import se_resnet50
import torch
from torch import nn
from torch.nn import functional as F
from segmentation_models_pytorch.base import modules as md
from typing import Optional, Union, List
# import sys
# sys.path.append('.')
from .modules import DropBlock2D
from .modules import BottleBlock
from .axial_attention_v2 import AxialAttentionBlock
from .model_util import get_lr_parameters
try:
    from inplace_abn import InPlaceABN
except:
    print("In order to use `use_batchnorm='inplace'` inplace_abn package must be installed. "
                + "To install see: https://github.com/mapillary/inplace_abn")
    

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
            drop_block_prob=0.1
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
            drop_block_prob=drop_block_prob
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
            drop_block_prob=drop_block_prob
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)


    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=None, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        activation = md.Activation(activation)
        super().__init__(conv2d, upsampling, activation)

class UnetPlusPlusDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
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
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type, drop_block_prob=drop_block_prob)

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
                blocks[f'x_{depth_idx}_{layer_idx}'] = DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)

        blocks[f'x_{0}_{len(self.in_channels)-1}'] = DecoderBlock(self.in_channels[-1], 0, self.out_channels[-1], **kwargs)
       
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


class UnetPlusPlusStar(nn.Module):
    def __init__(
            self,
            encoder_name,
            encoder_depth: int = 5,
            decoder_use_batchnorm: bool = True,
            decoder_channels: List[int] = (256, 128, 64, 32, 16),
            decoder_attention_type: Optional[str] = None,
            classes: int = 1,
            activation: Optional[Union[str, callable]] = None,
            deep_supervision=False,
            drop_block_prob=0.1
        ):
            super().__init__()
            self.encoder = get_encoder(encoder_name)

            self.decoder = UnetPlusPlusDecoder(
                encoder_channels=self.encoder.out_channels,
                decoder_channels=decoder_channels,
                n_blocks=encoder_depth,
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

            self.deep_supervision = deep_supervision
            self.name = f"unetplusplus-{encoder_name}"
            self.initialize()

    def initialize(self):
        initialize(self.encoder.layer4)
        initialize(self.decoder)
        initialize(self.segmentation_head)
        initialize(self.deep_segmentation_head)

    def forward(self, x):
        """Sequentially pass `x` trough model`s encoder, decoder and heads"""
        features = self.encoder(x)
        decoder_output = self.decoder(features)
        if self.deep_supervision:
            final_output = decoder_output[0]
            deep_outputs = decoder_output[1]
            final_mask = self.segmentation_head(final_output)
            masks = []
            for feature, segmentation_head in zip(deep_outputs, self.deep_segmentation_head):
                masks.append(segmentation_head(feature))
            return final_mask, masks
        else:
            final_output = decoder_output
            mask = self.segmentation_head(decoder_output)
            return mask

    def get_num_parameters(self):
        trainable= int(sum(p.numel() for p in self.parameters() if p.requires_grad))
        total = int(sum(p.numel() for p in self.parameters()))
        return trainable, total
    
    def get_paramgroup(self, base_lr=None):
        lr_dict = {
            "encoder.layer1": 0.1,
            "encoder.layer2": 0.1,
            "encoder.layer3": 0.1,
        }
        
        lr_group = get_lr_parameters(self, base_lr, lr_dict)
        return lr_group



if __name__ == '__main__':
    import torch.cuda.amp as amp
    model = UnetPlusPlusStar(
        encoder_name = 'BoTSER50', 
        decoder_attention_type='scse', 
        decoder_use_batchnorm='inplace', 
        deep_supervision=True, 
        drop_block_prob=0.1).cuda()
        
    print('Number params', model.get_num_parameters())
    img = torch.randn(2, 3, 1024, 1024).cuda()
    with amp.autocast():
        final_pred, preds = model(img) 
    for pred in preds:
        print(pred.shape)

    print(final_pred.shape)

    # optim = torch.optim.Adam(param_group, lr = 0.001, weight_decay=1e-5)
    # for param_group in optim.param_groups:
    #     print(float(param_group['lr']))
