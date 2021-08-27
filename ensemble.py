import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import ttach as tta
from pytorch_toolbelt.inference.tiles import ImageSlicer, TileMerger
from pytorch_toolbelt.utils.torch_utils import to_numpy, image_to_tensor, tensor_from_rgb_image

from catalyst.dl import utils
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import gc
import os
import json
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import subprocess

from src.main.aucpr import get_auc, plot_aucpr_curve
from src.data import NormalTransform
from src.data import TestSegmentation
from src.main.util import get_datapath, save_output as so
from src.main import archs

import logging
logging.basicConfig(level=logging.INFO)

def get_model(params, model_name):
    # Model return logit values
    params['encoder_weights'] = None
    model = getattr(smp, model_name)(
        **params
    )
    return model

def get_best_model(path):
    checkpoint = torch.load(path/'checkpoints/best.pth')
    with open(path / 'config.json', 'r') as j:
        config = json.load(j)

    if hasattr(smp, config['model_name']):
        model = get_model(
            config['model_params'], config['model_name'])
    elif config['model_name'] == "TransUnet":
        from self_attention_cv.transunet import TransUnet
        model = TransUnet(**config['model_params'])
    else:
        model = archs.get_model(
            model_name=config['model_name'], 
            params = config['model_params'],
            training=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model = model.to(utils.get_device())
    model = tta.SegmentationTTAWrapper(
            model, tta.aliases.d4_transform(), merge_mode="mean")

    return model

def predict(config, logdirs, outdir):
    test_img_dir = Path('data/raw/IDRiD/1. Original Images/b. Testing Set')
    test_mask_dir = Path('data/raw/IDRiD/2. All Segmentation Groundtruths/b. Testing Set')
    img_paths, mask_paths = get_datapath(test_img_dir, test_mask_dir, lesion_type=config['lesion_type'])

    models = []
    for logdir in logdirs:
       model = get_best_model(logdir)
       models.append(model)

    preprocessing_fn, _, _ = archs.get_preprocessing_fn(dataset_name=config['dataset_name'])
    transform = NormalTransform(config['scale_size'])
    augmentation = transform.resize_transforms() + [A.Lambda(image=preprocessing_fn), ToTensorV2()]
    test_transform = A.Compose(augmentation)
    test_ds = TestSegmentation(img_paths, mask_paths, transform=test_transform)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=2, pin_memory=True, shuffle=True)
    
    tta_predictions = []
    gt_masks = []
    filenames = []
    with torch.no_grad():
        for batch in test_loader:
            batch_imgs = batch['image'].to(utils.get_device())
            batch_gts = batch['mask']
            mean_pred = None
            for model in models:
                pred = model(batch_imgs)
                pred = pred.detach().cpu()
                pred = torch.sigmoid(pred)[0]
                pred = pred.squeeze(dim=0).numpy()
                if mean_pred is None: mean_pred = pred
                else: mean_pred += pred
            mean_pred = mean_pred / len(models)
            tta_predictions.append(mean_pred)
            gt_masks.append(batch_gts.numpy())
            filenames.append(batch['filename'][0])
    
    logging.info('====> Estimate auc-pr score')
    mean_auc = get_auc(gt_masks, tta_predictions, config)
    logging.info(f'MEAN-AUC {mean_auc}')
    logging.info('====> Find optimal threshold from 0 to 1 w.r.t auc-pr curve')
    optim_thres1, optim_thres2 = plot_aucpr_curve(
                                                gt_masks, 
                                                tta_predictions, 
                                                outdir, 
                                                config)
    
    logging.info(f"Optimal threshold is {optim_thres1}")
    logging.info('====> Output binary mask base on optimal threshold value')
    for mask_name, pred_mask in tqdm(zip(filenames, tta_predictions)):
        mask = (pred_mask > optim_thres1).astype(np.uint8)
        out_path = Path(config['out_dir']) / config['dataset_name'] / 'tta' / \
            config['lesion_type'] / outdir
        if not os.path.isdir(out_path):
            os.makedirs(out_path, exist_ok=True)

        out_name = out_path / mask_name
        so(mask, out_name)  # PIL Image format

    del gt_masks, tta_predictions, filenames
    gc.collect()
    logging.info('====> Finishing inference')


if __name__ == '__main__':
    subprocess.call(['wget', 'https://github.com/plotly/orca/releases/download/v1.2.1/orca-1.2.1-x86_64.AppImage', '-O', '/usr/local/bin/orca'])
    subprocess.call(['chmod', '+x', '/usr/local/bin/orca'])
    subprocess.call(['apt-get', 'install', 'xvfb', 'libgtk2.0-0', 'libgconf-2-4'])

    logdir_1 = Path('models/IDRiD/EX/') / 'Apr24_07_47'
    logdir_2 = Path('models/IDRiD/EX/') / 'Apr27_13_55'
    logdirs = [logdir_1, logdir_2]
    with open(logdir_1 / 'config.json', 'r') as j:
        config = json.load(j)

    config['out_dir'] = 'outputs'
    outdir = 'ensemble_1'
    predict(config, logdirs, outdir)
