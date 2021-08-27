import torch
import timm
from pprint import pprint

import sys
sys.path.append('..')

from archs.doubleunet import *

if __name__ == '__main__':
    print(__name__)
    m_lst = timm.list_models()
    pprint(m_lst)    
    device = 'cuda:0'
    encoder = Encoder1(backbone='mobilenetv3_large_100')
    model = Double_Unet(1, 0.2, encoder)
    model = model.to(device)

    a = torch.randn(1, 3, 256, 256, device = device)

    print(model)
    total = int(sum(p.numel() for p in model.parameters()))
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    count_parameters = {"total": total, "trainable": trainable}

    print(
        f'[INFO] total and trainable parameters in the model {count_parameters}')
    
    print(model(a).shape)