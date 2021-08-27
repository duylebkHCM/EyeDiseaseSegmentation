"""
@author: Duy Le <leanhduy497@gmail.com>
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import ttach as tta

from catalyst.dl import utils
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import albumentations.augmentations.crops.functional as F
import albumentations.augmentations.geometric.functional as GF

import os
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm

import rasterio
from rasterio.windows import Window

from torch.cuda import amp
import gc
import re

from .aucpr import get_auc, plot_aucpr_curve
from ..data import NormalTransform
from ..data import TestSegmentation
from .util import lesion_dict, get_datapath, make_grid, multigen ,save_output as so
from . import archs

import logging
logging.basicConfig(level=logging.INFO)

def get_model(params, model_name):
    # Model return logit values
    params['encoder_weights'] = None
    model = getattr(smp, model_name)(
        **params
    )
    return model

def str_2_bool(value: str):
    if value.lower() in ['1', 'true']:
        return True 
    elif value.lower() in ['0', 'false']:
        return False
    else:
        raise ValueError(f'Invalid value, should be one of these 1, true, 0, false')

def test_tta(logdir, config, args):
    test_img_dir = config['test_img_path']
    test_mask_dir = config['test_mask_path'] 
    img_paths, mask_paths = get_datapath(test_img_dir, test_mask_dir, lesion_type=config['lesion_type'])

    # Model return logit values
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
            
    preprocessing_fn, mean, std = archs.get_preprocessing_fn(dataset_name=config['dataset_name'], grayscale=config['gray'])

    transform = NormalTransform(config['scale_size'])
    augmentation = transform.resize_transforms() + [A.Lambda(image=preprocessing_fn), ToTensorV2()]

    test_transform = A.Compose(augmentation)

    test_ds = TestSegmentation(img_paths, config['gray'], mask_paths, transform=test_transform)
    ORI_H, ORI_W = test_ds.ori_h, test_ds.ori_w
    CROP_H, CROP_W = test_ds.crop_h, test_ds.crop_w
    test_loader = DataLoader(test_ds, batch_size=config['val_batch_size'], num_workers=2, pin_memory=True, shuffle=True)

    checkpoints = torch.load(f"{logdir}/checkpoints/{'best' if str_2_bool(args['best']) else 'last'}.pth")
    model.load_state_dict(checkpoints['model_state_dict'])
    model.eval()

    # D4 makes horizontal and vertical flips + rotations for [0, 90, 180, 270] angels.
    # and then merges the result masks with merge_mode="mean"
    tta_transform = getattr(tta.aliases, args['tta'] + '_transform')
    if args['tta'] == 'multiscale':
        param = {'scales': [1,2,4]}
        model = tta.SegmentationTTAWrapper(
            model, tta_transform(**param), merge_mode="mean")
    else:
        model = tta.SegmentationTTAWrapper(
            model, tta_transform(), merge_mode="mean")
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    model = model.to(utils.get_device())

    # this get predictions for the whole loader
    @multigen
    def predict_generator():
        with torch.no_grad():
            for batch in tqdm(test_loader):
                pred = model(batch['image'].to('cuda'))
                pred = pred.detach().cpu()
                pred = torch.sigmoid(pred)
                pred = pred.float().numpy()
                for i in range(len(batch['image'])):
                    crop_image = F.center_crop(pred[i][0], CROP_H, CROP_W)
                    crop_mask = F.center_crop(batch['mask'][i].numpy(), CROP_H, CROP_W)
                    image = GF.resize(crop_image, height=ORI_H, width=ORI_W, interpolation=cv2.INTER_LINEAR)
                    mask = GF.resize(crop_mask, height=ORI_H, width=ORI_W, interpolation=cv2.INTER_LINEAR)
                    yield image, mask, batch['filename'][i]

    logging.info('====> Estimate auc-pr score')
    mean_auc = get_auc(predict_generator(), config)
    logging.info(f'MEAN-AUC {mean_auc}')
    logging.info('====> Find optimal threshold from 0 to 1 w.r.t auc-pr curve')
    optim_thres1, optim_thres2, optim_thres3 = plot_aucpr_curve(
                                                predict_generator(),
                                                Path(logdir).name, 
                                                config)
    
    logging.info(f"Optimal threshold is {optim_thres3}")
    logging.info('====> Output binary mask base on optimal threshold value')
    # prob_dir =  Path(config['out_dir']) / config['dataset_name'] / 'tta' / \
    #         config['lesion_type'] / 'prob_image' / Path(logdir).name
    # threshold = args['optim_thres']  # Need to choose best threshold
    for pred_mask, _, mask_name in tqdm(predict_generator()):
        mask = (pred_mask > optim_thres3).astype(np.uint8)
        # print('Mask shape', mask.shape)
        out_path = Path(config['out_dir']) / config['dataset_name'] / 'tta' / \
            config['lesion_type'] / Path(logdir).name
        if not os.path.isdir(out_path):
            os.makedirs(out_path, exist_ok=True)

        out_name = out_path / mask_name
        so(mask, out_name)  # PIL Image format

    logging.info('====> Finishing inference')

def tta_patches(logdir, config, args):
    test_img_dir = config['test_img_path']
    test_mask_dir = config['test_mask_path'] / lesion_dict[config['lesion_type']].dir_name
    TEST_MASKS =  sorted(test_mask_dir.glob("*.*"))

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
    preprocessing_fn, _, _ = archs.get_preprocessing_fn(dataset_name=config['dataset_name'], grayscale=config['gray'])
    
    checkpoints = torch.load(f"{logdir}/checkpoints/{'best' if str_2_bool(args['best']) else 'last'}.pth")
    model.load_state_dict(checkpoints['model_state_dict'])
    model = model.to(utils.get_device())
    model.eval()

    tta_transform = getattr(tta.aliases, args['tta'] + '_transform')
    if args['tta'] == 'multiscale':
        param = {'scales': [1,2,4]}
        model = tta.SegmentationTTAWrapper(
            model, tta_transform(**param), merge_mode="mean")
    else:
        model = tta.SegmentationTTAWrapper(
            model, tta_transform(), merge_mode="mean")
    
    resize_size = config['scale_size']
    test_transform = A.Compose([
        A.Resize(resize_size, resize_size),
        A.Lambda(image = preprocessing_fn),
        ToTensorV2()
    ])

    def predict_generator():
        for mask_path in tqdm(TEST_MASKS):
            img = test_img_dir / re.sub('_' + config['lesion_type'] + '.tif', '.jpg', mask_path.name)
            gt_mask = Image.open(mask_path).convert('L')
            gt_mask = gt_mask.point(lambda x: 255 if x > 0 else 0, '1')
            gt_mask = np.asarray(gt_mask).astype(np.uint8)

            with rasterio.open(img.as_posix(), transform=rasterio.Affine(1, 0, 0, 0, 1, 0)) as dataset:
                slices = make_grid(dataset.shape, window=resize_size*2, min_overlap=32)
                preds = np.zeros(dataset.shape, dtype=np.float32)

                for (x1, x2, y1, y2) in slices:
                    image = dataset.read([1,2,3], window = Window.from_slices((x1, x2), (y1, y2)))
                    image = np.moveaxis(image, 0, -1)
                    image = test_transform(image=image)['image']
                    image = image.float()
                    
                    with torch.no_grad():
                        image = image.to(utils.get_device())[None]

                        logit = model(image)[0][0]
                        score_sigmoid = logit.sigmoid().cpu().numpy()
                        score_sigmoid = cv2.resize(score_sigmoid, (resize_size*2, resize_size*2), interpolation=cv2.INTER_LINEAR)

                        preds[x1:x2, y1:y2] = score_sigmoid
            
                yield preds, gt_mask, mask_path.name

    logging.info('====> Estimate auc-pr score')
    mean_auc = get_auc(predict_generator(), config)
    logging.info(f'MEAN-AUC {mean_auc}')
    logging.info('====> Find optimal threshold from 0 to 1 w.r.t auc-pr curve')
    optim_thres1, optim_thres2, optim_thres3 = plot_aucpr_curve(predict_generator(), Path(logdir).name, config)
    
    # prob_dir = Path(config['out_dir']) / config['dataset_name'] / 'tta' / \
    #                 config['lesion_type'] / 'prob_image' / Path(logdir).name
    for mask_pred, _, mask_name in tqdm(predict_generator()):
        mask = (mask_pred > optim_thres3).astype(np.float32)

        out_path = Path(config['out_dir']) / config['dataset_name'] / 'tta' / \
            config['lesion_type'] / Path(logdir).name

        if not os.path.isdir(out_path):
            os.makedirs(out_path, exist_ok=True)

        mask_name = re.sub('_' + config['lesion_type'] + '.tif', '.jpg', mask_name)
        out_name = out_path / mask_name
        so(mask, out_name)  # PIL Image format

    logging.info('====> Finishing inference')

if __name__ == '__main__':
    checkpoints = '../../models/IDRiD/SE/Apr26_09_24/checkpoints/best.pth'
    checkpoints = torch.load(checkpoints)
    model = archs.get_model()
    pass
