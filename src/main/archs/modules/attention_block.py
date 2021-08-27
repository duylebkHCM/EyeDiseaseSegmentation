#Credit: https://github.com/bigmb/Unet-Segmentation-Pytorch-Nest-of-Unets/blob/master/Models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Attention_block', 'SelfAttention', 'Spatial_Attention', 'Channel_Spatial_Attention']

class Attention_block(nn.Module):
    """
    Attention Block
    Paper: https://arxiv.org/abs/1804.03999
    """
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)   
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = x * psi
        return out

class SelfAttention(nn.Module):
    #Credit fast.ai
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels):
        super().__init__()
        self.query,self.key,self.value = [self._conv(n_channels, c) for c in (n_channels//8,n_channels//8,n_channels)]
        self.gamma = nn.Parameter(torch.tensor([0.]))

    def _conv(self,n_in,n_out):
        return nn.Conv2d(n_in, n_out, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        #Notation from the paper.
        size = x.size()
        x = x.view(*size[:2],-1)
        f,g,h = self.query(x),self.key(x),self.value(x)
        beta = F.softmax(torch.bmm(f.transpose(1,2), g), dim=1)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()

class AveragePool_Channel(nn.Module):
    """
    Average Pooling operation along channel axis
    """
    def __init__(self):
        super().__init__()
    def forward(self, input):
        output = torch.mean(input, dim=1, keepdim=True)
        assert output.shape[1] == 1, f'{output.shape}'
        return output

class MaxPool_Channel(nn.Module):
    """
    Max Pooling operation along channel axis
    """
    def __init__(self):
        super().__init__()
    def forward(self, input):
        _, output = torch.max(input, keepdim=True, dim=1)
        assert output.shape[1] == 1, f'{output.shape}'
        return output

class Spatial_Attention(nn.Module):
    """https://arxiv.org/abs/2004.03696"""
    def __init__(self):
        super().__init__()
        self.avg_pool = AveragePool_Channel()
        self.max_pool = MaxPool_Channel()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, stride=1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        f_avg = self.avg_pool(input)
        f_max = self.max_pool(input)
        concat = torch.cat([f_avg, f_max], dim=1)
        sa_map = self.conv(concat)
        output = sa_map*input
        return output

class Channel_Spatial_Attention(nn.Module):
    def __init__(self, channels, reduction, attention_kernel_size=3):
        super(Channel_Spatial_Attention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid_channel = nn.Sigmoid()

        self.conv_after_concat = nn.Conv2d(2, 1,
                                           kernel_size = attention_kernel_size,
                                           stride=1,
                                           padding = attention_kernel_size//2)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel attention module
        module_input = x
        avg = self.avg_pool(x)
        mx = self.max_pool(x)
        avg = self.fc1(avg)
        mx = self.fc1(mx)
        avg = self.relu(avg)
        mx = self.relu(mx)
        avg = self.fc2(avg)
        mx = self.fc2(mx)
        x = avg + mx
        x = self.sigmoid_channel(x)

        # Spatial attention module
        x = module_input * x
        module_input = x

        avg = torch.mean(x, 1, True)
        mx, _ = torch.max(x, 1, True)
        x = torch.cat((avg, mx), 1)
        x = self.conv_after_concat(x)
        x = self.sigmoid_spatial(x)
        x = module_input * x
        return x

if __name__ == '__main__':
    layer = Spatial_Attention()
    a = torch.randn(1, 10, 64, 64)

    print(layer(a).shape)