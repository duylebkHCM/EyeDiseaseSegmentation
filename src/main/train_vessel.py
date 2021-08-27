"""
@author: Duy Le <leanhduy497@gmail.com>
"""
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import torch.nn as nn
from pytorch_toolbelt.utils.catalyst import (
    HyperParametersCallback,
    draw_binary_segmentation_predictions,
    ShowPolarBatchesCallback,
)
from catalyst.contrib.nn import OneCycleLRWithWarmup
from catalyst import dl, metrics
from catalyst.contrib.callbacks.wandb_logger import WandbLogger
from catalyst.dl import SupervisedRunner, CriterionCallback, EarlyStoppingCallback, SchedulerCallback, MetricAggregationCallback, IouCallback, DiceCallback
from catalyst import utils
from functools import partial
from collections import OrderedDict
from pathlib import Path
from typing import List, Union, Tuple
import os
import json
import logging
import glob
logging.basicConfig(level=logging.INFO, format='')

from . import util
from .util import lesion_dict
from ..data import *
from .scheduler import get_scheduler
from .optim import get_optimizer
from .losses import get_loss, WeightedBCEWithLogits
from ..data.vessel_dataset import  OneLesionSegmentation
from . import archs

def get_model(params, model_name):
    
    # Model return logit values
    model = getattr(smp, model_name)(
        **params
    )

    return model

def get_loader(
    images: Union[List[str], Tuple[List[str]]],
    is_gray: bool,
    random_state: int,
    masks: Union[List[str], Tuple[List[str]]] = None,
    valid_size: float = 0.2,
    batch_size: int = 4,
    val_batch_size: int = 8,
    num_workers: int = 4,
    train_transforms_fn=None,
    valid_transforms_fn=None,
    preprocessing_fn=None,
    ben_method = None,
    mode='binary',
    data_type = 'tile'
):  
    if isinstance(images, List):
        if data_type == 'all':
            indices = np.arange(len(images))

            train_indices, valid_indices = train_test_split(
                indices, test_size=valid_size, random_state=random_state, shuffle=True)

            np_images = np.array(images)
            train_imgs = np_images[train_indices].tolist()
            valid_imgs = np_images[valid_indices].tolist()
        else:
            train_df = pd.read_csv('data/processed/IDRiD/train/SE/img_mask.csv')
            train_df = train_df.sample(frac=1).reset_index(drop=True)
            train_imgs= train_df.loc[:, 'img'].values.tolist()
            train_imgs = [Path('/'.join(list(Path(path).parts[2:])) + '/') for path in train_imgs]
            train_masks = train_df.loc[:, 'mask'].values.tolist()
            train_masks = [Path('/'.join(list(Path(path).parts[2:])) + '/') for path in train_masks]
            
            val_df = pd.read_csv('data/processed/IDRiD/val/SE/img_mask.csv')
            val_df = val_df.sample(frac=1).reset_index(drop=True)
            valid_imgs= val_df.loc[:, 'img'].values.tolist()
            valid_imgs = [Path('/'.join(list(Path(path).parts[2:])) + '/') for path in valid_imgs]
            valid_masks = val_df.loc[:, 'mask'].values.tolist()
            valid_masks = [Path('/'.join(list(Path(path).parts[2:])) + '/') for path in valid_masks]
    else:
        if data_type == 'all':
            train_imgs = images[0]
            valid_imgs = images[1]
        else:
            pass
        
    if mode == 'binary':
        if isinstance(masks, List):
            if data_type == 'all':
                np_masks = np.array(masks)
                train_masks = np_masks[train_indices].tolist()
                valid_masks = np_masks[valid_indices].tolist()
            else:
                pass
        else:
            if data_type == 'all':
                train_masks = masks[0]
                valid_masks = masks[1]
            else:
                pass
            
        train_dataset = OneLesionSegmentation(
            train_imgs,
            is_gray,
            masks=train_masks,
            transform=train_transforms_fn,
            preprocessing_fn=preprocessing_fn,
            ben_transform = ben_method,
            data_type = data_type
        )

        valid_dataset = OneLesionSegmentation(
            valid_imgs,
            is_gray,
            masks=valid_masks,
            transform=valid_transforms_fn,
            preprocessing_fn=preprocessing_fn,
            ben_transform = ben_method,
            data_type = data_type
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=val_batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        drop_last=False
    )

    loaders = OrderedDict()
    loaders['train'] = train_loader
    loaders['valid'] = valid_loader

    log_info = [['train', 'valid'], [[len(train_loader), len(valid_loader)], [
        len(train_dataset), len(valid_dataset)]]]

    return loaders, log_info


def train_model(exp_name, configs, seed):
    torch.autograd.set_detect_anomaly(True)

    TRAIN_IMG_DIRS = str(configs['train_img_path'])
    TRAIN_MASK_DIRS = str(configs['train_mask_path'])
    VAL_IMG_DIRS = str(configs['val_img_path'])
    VAL_MASK_DIRS = str(configs['val_mask_path']
)
    # Get model
    use_smp = True
    if hasattr(smp, configs['model_name']):
        model = get_model(
            configs['model_params'], configs['model_name'])
    elif configs['model_name'] == "TransUnet":
        from self_attention_cv.transunet import TransUnet
        model = TransUnet(**configs['model_params'])
        use_smp=False
    else:
        model = archs.get_model(
            model_name=configs['model_name'], 
            params = configs['model_params'])
        use_smp = False

    preprocessing_fn, mean, std = archs.get_preprocessing_fn(dataset_name=None, grayscale=configs['gray'])

    #Define transform (augemntation)
    # Transform = get_transform(configs['augmentation'])
    # transforms = Transform(
    #     configs['scale_size'],
    #     preprocessing_fn=preprocessing_fn
    # )

    # train_transform = transforms.train_transform()
    # val_transform = transforms.validation_transform()
    # preprocessing = transforms.get_preprocessing()

    # if configs['data_mode'] == 'binary':
    if configs['dataset_name'] == 'DRIVE':
        ext = '/*.png'
    else:
        ext = '/*.jpg'

    ex_dirs, mask_dirs = (sorted(glob.glob(TRAIN_IMG_DIRS + ext)), sorted(glob.glob(VAL_IMG_DIRS + ext))), (sorted(glob.glob(TRAIN_MASK_DIRS + '/*.png')), sorted(glob.glob(VAL_MASK_DIRS + '/*.png')))

    # util.log_pretty_table(['full_img_paths', 'full_mask_paths'], [
    #                             [len(ex_dirs), len(mask_dirs)]])
    # elif configs['data_mode'] == 'multiclass':
    #     pass
    # else:
    #     ex_dirs = list(TRAIN_IMG_DIRS.glob('*.jpg'))
    #     mask_dirs = None

    # Get data loader
    if configs['use_ben_transform']:
        ben_transform = load_ben_color
    else:
        ben_transform = None

    loader, log_info = get_loader(
        images=ex_dirs,
        is_gray=configs['gray'],
        masks=mask_dirs,
        random_state=seed,
        batch_size=configs['batch_size'],
        val_batch_size=configs['val_batch_size'],
        num_workers=0,
        train_transforms_fn=None,
        valid_transforms_fn=None,
        preprocessing_fn=preprocessing_fn,
        ben_method = ben_transform,
        mode=configs['data_mode'],
        data_type=configs['data_type']
    )

    #Visualize on terminal
    util.log_pretty_table(log_info[0], log_info[1])

    if use_smp:
        if configs['finetune']:
            #Free all weights in the encoder of model
            for param in model.encoder.parameters():
                param.requires_grad = False
        if configs['model_params']['encoder_weights'] is not None:
            bn_types = nn.BatchNorm2d
            #Disable batchnorm update
            for m in model.encoder.modules():
                if isinstance(m, bn_types):
                    m.eval()

    param_group = model.get_paramgroup(configs['learning_rate_decode'], weight_decay=configs['weight_decay'])
    trainable, total = model.get_num_parameters()
    count_parameters = {"total": total, "trainable": trainable}

    logging.info(
        f'[INFO] total and trainable parameters in the model {count_parameters}')
    
    #Set optimizer
    optimizer = get_optimizer(
        configs['optimizer'], param_group, configs['learning_rate_decode'], configs['weight_decay'])
    #Set learning scheduler
    scheduler = get_scheduler(
        configs['scheduler'], optimizer, configs['learning_rate'], configs['num_epochs'],
        batches_in_epoch=len(loader['train']), mode=configs['mode']
    )
    #Set loss
    criterion = {}
    for loss_name in configs['criterion']:
        if loss_name == 'wbce':
            pos_weights = torch.tensor(configs['pos_weights'], device=utils.get_device())
            loss_fn = WeightedBCEWithLogits(pos_weights=pos_weights)
        else:
            loss_fn = get_loss(loss_name)
        criterion[loss_name] = loss_fn

    #Define callbacks
    callbacks = []
    losses = []
    for loss_name, loss_weight in configs['criterion'].items():
        criterion_callback = CriterionCallback(
            input_key="mask",
            output_key="logits",
            criterion_key=loss_name,
            prefix="loss_"+loss_name,
            multiplier=float(loss_weight)
        )

        callbacks.append(criterion_callback)
        losses.append(criterion_callback.prefix)

    callbacks += [MetricAggregationCallback(
        prefix="loss",
        mode="sum",
        metrics=losses
    )]

    if isinstance(scheduler, (CyclicLR, OneCycleLRWithWarmup)):
        callbacks += [SchedulerCallback(mode="batch")]
    elif isinstance(scheduler, (ReduceLROnPlateau)):
        callbacks += [SchedulerCallback(reduced_metric=configs['metric'])]

    hyper_callbacks = HyperParametersCallback(configs)

    if configs['data_mode'] == 'binary':
        image_format = 'gray' if configs['gray'] else 'rgb'
        visualize_predictions = partial(
            draw_binary_segmentation_predictions, image_key="image", targets_key="mask", mean=mean, std=std, image_format=image_format
        )
    # elif configs['data_mode'] == 'multilabel':
    #     visualize_predictions = partial(
    #         draw_multilabel_segmentation_predictions, image_key="image", targets_key="mask", class_colors=CLASS_COLORS
    #     )

    show_batches_1 = ShowPolarBatchesCallback(
        visualize_predictions, metric="iou", minimize=False)

    # show_batches_2 = ShowPolarBatchesCallback(
    #     visualize_predictions, metric="loss", minimize=True)

    early_stopping = EarlyStoppingCallback(
        patience=20, metric=configs['metric'], minimize=False)

    iou_scores = IouCallback(
        input_key="mask",
        activation="Sigmoid",
        threshold=0.5
    )

    dice_scores = DiceCallback(
        input_key="mask",
        activation="Sigmoid",
        threshold=0.5
    )

    # aucpr_scores = AucPRMetricCallback(
    #     input_key="mask",
    # )

    # aucroc_scores = RocAucMetricCallback(
    #     input_key="mask",
    # )
    #End define
    # if configs['resume'] is not None:
    #     exp_name = configs['resume'].split(os.path.sep)[-3]

    prefix = f"{configs['lesion_type']}/{exp_name}"
    log_dir = os.path.join("models/", configs['dataset_name'], prefix)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # if configs['kfold']:
    #     logger = WandbLogger(project=lesion_dict[configs['lesion_type']].project_name,
    #                      name=)

    logger = WandbLogger(project=lesion_dict[configs['lesion_type']].project_name,
                         name=exp_name)


    #Save config as JSON format
    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        configs['train_img_path'] = str(configs['train_img_path'])
        configs['train_mask_path'] = str(configs['train_mask_path'])
        configs['val_img_path'] = str(configs['val_img_path'])
        configs['val_mask_path'] = str(configs['val_mask_path'])
        json.dump(configs, f)

    callbacks += [hyper_callbacks, early_stopping,
                  iou_scores, dice_scores, show_batches_1, logger]

    
    # class CustomRunner(dl.SupervisedRunner):
    #     def _handle_batch(self, batch):
    #         x, y = batch

    #         y_pred = self.model(x)
    #         pass

    if configs['is_fp16']:
        fp16_params = dict(amp=True)  # params for FP16
    else:
        fp16_params = None
        
    # model training
    if not configs['deep_supervision']:
        runner = SupervisedRunner(
            device=utils.get_device(), input_key="image", input_target_key="mask")

        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            callbacks=callbacks,
            logdir=log_dir,
            loaders=loader,
            num_epochs=configs['num_epochs'],
            scheduler=scheduler,
            main_metric=configs['metric'],
            minimize_metric=False,
            timeit=True,
            fp16=fp16_params,
            resume=configs['resume_path'],
            verbose=True,
        )
    else:
        callbacks = []
        if isinstance(scheduler, (CyclicLR, OneCycleLRWithWarmup)):
            callbacks += [SchedulerCallback(mode="batch")]
        elif isinstance(scheduler, (ReduceLROnPlateau)):
            callbacks += [SchedulerCallback(reduced_metric=configs['metric'])]

        hyper_callbacks = HyperParametersCallback(configs)

        optim_cb =  dl.OptimizerCallback(
                    metric_key="loss",    
                    accumulation_steps=1,  
                    grad_clip_params=None
                )
        
        callbacks += [optim_cb, hyper_callbacks, early_stopping, logger]

        def get_pyramid(mask: torch.Tensor, height, shape_list, include_final_mask=False):
            with torch.no_grad():
                if include_final_mask:
                    masks = [mask]
                    big_mask = masks[-1]
                else:
                    masks = []
                    big_mask = mask
                for _, shape in zip(range(height), shape_list):
                    small_mask = F.adaptive_avg_pool2d(big_mask, shape)
                    masks.append(small_mask)
                    big_mask = masks[-1]

                targets = []
                for mask in masks:
                    targets.append(mask)

            return targets

        class CustomRunner(dl.Runner):
            def _handle_batch(self, batch):
                results = batch
                x = results['image']
                y = results['mask']
                y_hat, y_hat_levels = self.model(x)
                shape_list =[]
                for level in y_hat_levels:
                    shape_list.append(np.array(level.shape[2:]).tolist())

                targets = get_pyramid(y, len(y_hat_levels), shape_list, False)
                loss_levels = []

                criterion_ds = get_loss(configs['criterion_ds'])
                for y_hat_level, target in zip(y_hat_levels, targets):
                    loss_levels.append(criterion_ds(y_hat_level, target))

                loss_final = None  
                loss_com = {}
                for loss_name, loss_weight in configs['criterion'].items():
                    loss_com['loss_' + loss_name] = criterion[loss_name](y_hat, y) 
                    if loss_final is None:
                        loss_final = criterion[loss_name](y_hat, y)*float(loss_weight)
                    else:
                        loss_final += criterion[loss_name](y_hat,y)*float(loss_weight)

                loss_deep_super = torch.sum(torch.stack(loss_levels))
                loss = loss_final + loss_deep_super

                target = y
                pred = torch.sigmoid(y_hat)
                dice  = metrics.dice(pred, target, threshold=0.5).mean()
                iou = metrics.iou(pred, target, threshold=0.5).mean()

                self.batch_metrics = {
                    'loss': loss,
                    'dice': dice,
                    'iou': iou
                }

                for key in loss_com:
                    self.batch_metrics[key] = loss_com[key]
            
        runner = CustomRunner(device = utils.get_device())
        runner.train(
            model=model,
            optimizer=optimizer,
            callbacks=callbacks,
            logdir=log_dir,
            loaders=loader,
            num_epochs=configs['num_epochs'],
            scheduler=scheduler,
            main_metric=configs['metric'],
            minimize_metric=False,
            timeit=True,
            fp16=fp16_params,
            resume=configs['resume_path'],
            verbose=True,
        )

def run_cross_validation(fold):
    #@TODO Not doing yet because training with cross validation take so much time 
    pass
