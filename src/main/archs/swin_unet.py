import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


# import sys
# sys.path.append('.')

from .modules.swin_transformer import SwinTransformerBlock, PatchMerging, PatchEmbed, SwinTransformer, BasicLayer


class Encoder(nn.Module):
    def __init__(self):
        encoder = SwinTransformer()
    def forward(self, x):

        pass

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        return x



class PatchMerging(nn.Module):
    """ Patch Merging Layer

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpanding(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)


        pass
    pass

class Decoder(nn.Module):
    pass


if __name__ == '__main__':
    #Patch embedding input images
    pad_embed = PatchEmbed()
    a = torch.rand(2, 3, 1024, 1024)
    out = pad_embed(a)

    print(out.shape) # shape 2, 96, 256, 256

    Wh, Ww = out.size(2), out.size(3)
    pretrained_img_size=224

    patch_grid = [pretrained_img_size // 4, pretrained_img_size //4]
    absolute_pos_embed = nn.Parameter(torch.zeros(1, 96, patch_grid[0], patch_grid[1]))
    trunc_normal_(absolute_pos_embed, std=0.02)
    print(absolute_pos_embed.shape)

    absolute_pos_embed = F.interpolate(absolute_pos_embed, size=(Wh, Ww), mode='bicubic', align_corners=False)
    print(absolute_pos_embed.shape)

    out = (out + absolute_pos_embed)
    print(out.shape)

    out = out.flatten(2)

    print(out.shape)

    out = out.transpose(1, 2)
    print(out.shape)

    depth = 4

    # blocks = nn.ModuleList([
    #     SwinTransformerBlock(
    #         dim=dim,
    #         num_heads=num_heads,
    #         window_size=window_size,
    #         shift_size=0 if (i % 2 == 0) else window_size // 2,
    #         mlp_ratio=mlp_ratio,
    #         qkv_bias=qkv_bias,
    #         qk_scale=qk_scale,
    #         drop=drop,
    #         attn_drop=attn_drop,
    #         drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
    #         norm_layer=norm_layer)
    #     for i in range(depth)])

    pad_merge = PatchMerging(96)
    H, W = 256, 256
    merge = pad_merge(out, H, W)

    print(merge.shape)