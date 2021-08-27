import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    """
    Split images into patches and then embed them

    Parameters:
    ----------
    img_size: int
        Size of the image (it is a square).
    
    patch_size: int
        Size of the patch (it is a square).

    in_chans: int
        Number of input channels
    
    embed_dims: int
        The embedding dimensions
    """

    