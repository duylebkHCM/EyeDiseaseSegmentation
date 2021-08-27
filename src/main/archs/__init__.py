import torch.nn as nn
import numpy as np
from . import attentionunet, hrnet, doubleunet, dbunet, unet, rcnn_unet, sa_unet, hed, fpn, unets, deeplab, transunet, unetplusplusstar, LeeJunHyun_impl, unet3plus, axial_attentionunet, dcunet, resunetplusplus, deep_supunetplusplus, hubmap_kaggle, deep_supdeeplabv3plus, transunetv2, segformerstar, swinformerstar

__all__ = ['list_models', 'get_model', 'get_preprocessing_fn']

MODEL_REGISTRY = {
    "resnet50_attunet": attentionunet.resnet50_attunet, 
    "seresnet50_attunet":attentionunet.seresnet50_attunet,
    "efficientnetb2_attunet": attentionunet.efficientnetb2_attunet,
    "mobilenetv3_attunet": attentionunet.mobilenetv3_attunet,
    "seresnet50_attunet": attentionunet.seresnet50_attunet,
    "swin_tiny_attunet": attentionunet.swin_tiny_attunet,
    "swin_small_attunet": attentionunet.swin_small_attunet,
    "hrnet18": hrnet.hrnet18, 
    "hrnet34": hrnet.hrnet34, 
    "hrnet48": hrnet.hrnet48,
    "resnet50_doubleunet": doubleunet.resnet50_doubleunet,
    "efficientnetb2_doubleunet": doubleunet.efficientnetb2_doubleunet,
    "mobilenetv3_doubleunet": doubleunet.mobilenetv3_doubleunet,
    "vgg_doubleunet": dbunet.DUNet,
    "unet_resnext50_ssl": unet.UneXt50,
    "rrcnn_unet": rcnn_unet.R2U_Net,
    "sa_unet": sa_unet.sa_unetbase,
    "hed_unet": hed.hed_unet ,
    "hed_resunet": hed.hed_resunet,
    "hed_denseunet": hed.hed_denseunet,
    "resnet18_unet32": unets.resnet18_unet32,
    "resnet34_unet32": unets.resnet34_unet32,
    "resnet50_unet32": unets.resnet50_unet32,
    "b4_unet32":unets.b4_unet32,
    "b4_effunet32": unets.b4_effunet32,
    "b2_effunet32": unets.b2_effunet32,
    "b2_fpn_cat": fpn.b2_fpn_cat,
    "seresnext50_fpncat128": fpn.seresnext50_fpncat128,
    "resnet34_fpncat128": fpn.resnet34_fpncat128,
    "resnet152_fpncat256": fpn.resnet152_fpncat256,
    "transunet_r50": transunet.TransUnet_R50,
    "transunet_b16": transunet.TransUnet_B16,
    "unetplusplusstar": unetplusplusstar.UnetPlusPlusStar,
    "LeeJunHyun_impl_att": LeeJunHyun_impl.AttU_Net,
    "LeeJunHyun_impl_R2U_Net": LeeJunHyun_impl.R2U_Net,
    "LeeJunHyun_impl_R2AttU_Net": LeeJunHyun_impl.R2AttU_Net ,
    "Unet3Plus_Base": unet3plus.UNet_3Plus,
    "Unet3Plus_DS": unet3plus.UNet_3Plus_DeepSup,
    "axialatt_unet": axial_attentionunet.axialunet,
    "gated": axial_attentionunet.gated,
    "medt": axial_attentionunet.MedT,
    "logo": axial_attentionunet.logo,
    "axialattwopo_unet": axial_attentionunet.axialunet_wopo,
    "dcunet": dcunet.DcUnet,
    "resunetplusplus": resunetplusplus.ResUnetPlusPlus,
    "unetplusplus_deepsup": deep_supunetplusplus.UnetPlusPlus,
    "hubmap_kaggle": hubmap_kaggle.UNET_SERESNEXT101,
    "deeplabv3plus_deepsup": deep_supdeeplabv3plus.DeepLabV3Plus,
    "TransUnet_V2": transunetv2.TransUnet,
    "SegFormerStar": segformerstar.SegformerStar,
    "SwinformerStar": swinformerstar.SwinformerStar
}

def get_preprocessing_fn(dataset_name: str, grayscale: bool):
    if dataset_name == "IDRiD":
        mean = [0.44976714,0.2186806,0.06459363]
        std = [0.33224553,0.17116262,0.086509705]
    elif dataset_name == 'FGADR':
        mean = [0.4554011,0.2591345,0.13285689]
        std = [0.28593522,0.185085,0.13528904]
    elif dataset_name == 'DDR':
        mean = [0.31897065,0.19916488,0.08322998]
        std = [0.32040685,0.20822203,0.114768185]
    elif dataset_name == 'DRIVE':
        mean = [0.49742976,0.27066445,0.16217253]
        std = [0.34794736,0.18998094,0.1084089]
    elif dataset_name == 'HRF':
        mean = [0.6273858,0.20169912,0.10424815]
        std = [0.2866019,0.11408445,0.060513902]
    elif dataset_name == 'CHASEDB1':
        mean = [0.4527923,0.16221291,0.028265305]
        std = [0.36041078,0.14167951,0.036878455]
    else:
        mean = [0.44976714,0.2186806,0.06459363]
        std = [0.33224553,0.17116262,0.086509705]

    if grayscale:
        mean = mean[0]*0.2989 + mean[1]*0.5870 +mean[2]*0.1140
        std = std[0]*0.2989 +std[1]*0.5870 +std[2]*0.1140
        
    def preprocessing(x, mean=mean, std=std, **kwargs):
        x = x / 255.0
        if mean is not None:
            mean = np.array(mean)
            x = x - mean

        if std is not None:
            std = np.array(std)
            x = x / std
        return x

    return preprocessing, mean, std

def list_models():
    return list(MODEL_REGISTRY.keys())

def get_model(model_name: str, params=None, training=True) -> nn.Module:   
    try:
        model_fn = MODEL_REGISTRY[model_name]
    except KeyError:
        raise KeyError(f"Cannot found {model_name}, available options are {list(MODEL_REGISTRY.keys())}")
    if params is None:
        return model_fn()
    if not training:
        if params.get('clfhead', None) is not None:
            params['clfhead'] = False
        if params.get('pretrained', None) is not None:
            params['pretrained'] = False
        if params.get('encoder_weights', None) is not None:
            params['encoder_weights'] = None
        if params.get('deep_supervision', None) is not None:
            params['deep_supervision'] = False
    return model_fn(**params)
