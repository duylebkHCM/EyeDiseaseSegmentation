import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, input_channels:list, output_channels:list):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                nn.Sequential(nn.Conv2d(in_ch, out_ch*2, kernel_size=3, padding=1),
                nn.ReLU(inplace=True), nn.BatchNorm2d(out_ch*2),
                nn.Conv2d(out_ch*2, out_ch, kernel_size=3, padding=1))
                for in_ch, out_ch in zip(input_channels, output_channels)
            ]
        )
        
    def forward(self, xs:list, last_layer):
        hcs = [F.interpolate(c(x),scale_factor=2**(len(self.convs)-i),mode='bilinear') 
               for i,(c,x) in enumerate(zip(self.convs, xs))]
        hcs.append(last_layer)
        return torch.cat(hcs, dim=1)