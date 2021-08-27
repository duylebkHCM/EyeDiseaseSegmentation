import argparse
from catalyst.contrib.callbacks import DrawMasksCallback
from catalyst.dl import DiceCallback, IouCallback, CriterionCallback, MetricAggregationCallback
from catalyst.dl import SupervisedRunner
import torch.optim as optim
import torch.nn as nn
from catalyst.contrib.nn import DiceLoss, IoULoss
import segmentation_models_pytorch as smp
import collections
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from typing import List
import numpy as np
from pathlib import Path
import os
import torch
import catalyst
from catalyst import utils
import ttach as tta

from base_dataset import LesionSegmentation
from base_transform import Transform
from base_utils import get_datapath, save_output as so


def get_loader(
    images: List[Path],
    masks: List[Path],
    random_state: int,
    valid_size: float = 0.2,
    batch_size: int = 4,
    num_workers: int = 4,
    train_transforms_fn=None,
    valid_transforms_fn=None,
):
    indices = np.arange(len(images))

    train_indices, valid_indices = train_test_split(
        indices, test_size=valid_size, random_state=random_state, shuffle=True)

    np_images = np.array(images)
    np_masks = np.array(masks)

    train_dataset = LesionSegmentation(
        np_images[train_indices].tolist(),
        np_masks[train_indices].tolist(),
        transform=train_transforms_fn
    )

    valid_dataset = LesionSegmentation(
        np_images[valid_indices].tolist(),
        np_masks[valid_indices].tolist(),
        transform=valid_transforms_fn
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True
    )

    loaders = collections.OrderedDict()
    loaders['train'] = train_loader
    loaders['valid'] = valid_loader

    return loaders


def main(args):
    DEVICE = utils.get_device()
    if torch.cuda.is_available():
        print('GPU is available')
        print(f'Number of available gpu: {torch.cuda.device_count()} GPU')
        print(f'GPU: {DEVICE}')
    else:
        print('Oops! sorry dude, it`s seem like you have to use CPU instead :))')

    is_fp16_used = args['fp16']

    if is_fp16_used:
        batch_size = 4
    else:
        batch_size = 2

    ENCODER = 'resnet18'
    ENCODER_WEIGHTS = 'imagenet'

    # Model return logit values
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        activation=None,
        in_channels=3,
        classes=1
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(
        ENCODER, ENCODER_WEIGHTS)

    img_paths, mask_paths = get_datapath(
        train_image_dir, train_mask_dir, lesion_type=args['type'])

    transforms = Transform(1024, preprocessing_fn)
    train_transform = transforms.train_transform()
    valid_transform = transforms.validation_transform()

    loaders = get_loader(
        images=img_paths,
        masks=mask_paths,
        random_state=SEED,
        batch_size=batch_size,
        train_transforms_fn=train_transform,
        valid_transforms_fn=valid_transform
    )

    criterion = {
        'dice': DiceLoss(),
        'iou': IoULoss(),
        'bce': nn.BCEWithLogitsLoss()
    }

    learning_rate = args['lr']
    num_epochs = args['epochs']

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.25, patience=2)

    logdir = os.path.join(
        CKPT, args['type']+'_lr' + str(args['lr']) + '_ep' + str(args['epochs']))

    if not os.path.isdir(logdir):
        os.makedirs(logdir, exist_ok=True)

    if is_fp16_used:
        fp16_params = dict(opt_level="O1")  # params for FP16
    else:
        fp16_params = None

    print(f"FP16 params: {fp16_params}")

    # by default SupervisedRunner uses "features" and "targets",
    # in our case we get "image" and "mask" keys in dataset __getitem__
    runner = SupervisedRunner(
        device=DEVICE, input_key="image", input_target_key="mask")

    callbacks = [
        # Each criterion is calculated separately.
        CriterionCallback(
            input_key="mask",
            prefix="loss_dice",
            criterion_key="dice"
        ),
        CriterionCallback(
            input_key="mask",
            prefix="loss_iou",
            criterion_key="iou"
        ),
        CriterionCallback(
            input_key="mask",
            prefix="loss_bce",
            criterion_key="bce"
        ),

        # And only then we aggregate everything into one loss.
        MetricAggregationCallback(
            prefix="loss",
            mode="weighted_sum",  # can be "sum", "weighted_sum" or "mean"
            # because we want weighted sum, we need to add scale for each loss
            metrics={"loss_dice": 1.0, "loss_iou": 1.0, "loss_bce": 0.8},
        ),

        # metrics
        DiceCallback(input_key="mask"),
        IouCallback(input_key="mask"),
        # visualization
        DrawMasksCallback(output_key='logits',
                          input_image_key='image',
                          input_mask_key='mask',
                          summary_step=50
                          )
    ]

    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        # our dataloaders
        loaders=loaders,
        # We can specify the callbacks list for the experiment;
        callbacks=callbacks,
        # path to save logs
        logdir=logdir,
        num_epochs=num_epochs,
        # save our best checkpoint by IoU metric
        main_metric="iou",
        # IoU needs to be maximized.
        minimize_metric=False,
        # for FP16. It uses the variable from the very first cell
        fp16=fp16_params,
        # prints train logs
        verbose=True,
    )

    return {'runner': runner, 'transform': transforms, 'logdir': logdir}


def test(args, info: dict):
    runner = info['runner']
    transform = info['transform']
    logdir = info['logdir']

    TEST_IMAGES = sorted(test_image_dir.glob("*.jpg"))

    test_transform = transform.test_transform()
    # create test dataset
    test_dataset = LesionSegmentation(
        TEST_IMAGES,
        transform=test_transform
    )

    num_workers: int = 4

    infer_loader = DataLoader(
        test_dataset,
        batch_size=args['testbatchsize'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # this get predictions for the whole loader
    predictions = np.vstack(list(map(
        lambda x: x["logits"].cpu().numpy(),
        runner.predict_loader(loader=infer_loader,
                              resume=f"{logdir}/checkpoints/best.pth", fp16=args['fp16'])
    )))

    threshold = 0.5

    for i, (features, logits) in enumerate(zip(test_dataset, predictions)):
        image_name = features['filename']
        mask_ = torch.from_numpy(logits[0]).sigmoid()
        mask = utils.detach(mask_ > threshold).astype("float")

        out_path = Path(OUTDIR) / Path(logdir).name
        if not os.path.isdir(out_path):
            os.makedirs(out_path, exist_ok=True)

        out_name = out_path / image_name
        so(mask, out_name)  # PIL Image format


def test_tta(args, info: dict):
    transform = info['transform']
    logdir = info['logdir']

    TEST_IMAGES = sorted(test_image_dir.glob("*.jpg"))

    test_transform = transform.test_transform()
    # create test dataset
    test_dataset = LesionSegmentation(
        TEST_IMAGES,
        transform=test_transform
    )

    num_workers: int = 4

    infer_loader = DataLoader(
        test_dataset,
        batch_size=args['testbatchsize'],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    ENCODER = 'resnet18'

    # Model return logit values
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=None,
        activation=None,
        in_channels=3,
        classes=1
    )

    checkpoints = torch.load(f"{logdir}/checkpoints/best.pth")
    model.load_state_dict(checkpoints['model_state_dict'])

    # D4 makes horizontal and vertical flips + rotations for [0, 90, 180, 270] angels.
    # and then merges the result masks with merge_mode="mean"
    tta_model = tta.SegmentationTTAWrapper(
        model, tta.aliases.d4_transform(), merge_mode="mean")

    tta_runner = SupervisedRunner(
        model=tta_model,
        device=utils.get_device(),
        input_key="image"
    )

    # this get predictions for the whole loader
    tta_predictions = [] 
    for batch in infer_loader:
        tta_pred = tta_runner.predict_batch(batch)
        # print(tta_pred['logits'].cpu().numpy().shape)
        tta_predictions.append(tta_pred['logits'].cpu().numpy())
    
    tta_predictions = np.vstack(tta_predictions)
    # print(tta_predictions.shape)

    threshold = 0.5

    # import sys
    # sys.exit(1)

    for i, (features, logits) in enumerate(zip(test_dataset, tta_predictions)):
        image_name = features['filename']
        mask_ = torch.from_numpy(logits[0]).sigmoid()
        mask = utils.detach(mask_ > threshold).astype("float")

        out_path = Path(TTA_DIR) / Path(logdir).name
        if not os.path.isdir(out_path):
            os.makedirs(out_path, exist_ok=True)

        out_name = out_path / image_name
        so(mask, out_name)  # PIL Image format


if __name__ == '__main__':
    parse = argparse.ArgumentParser()

    parse.add_argument('--type', required=True, type=str)
    parse.add_argument('--fp16', default=True, type=bool)
    parse.add_argument('--lr', default=0.001, type=float)
    parse.add_argument('--epochs', default=20, type=int)
    parse.add_argument('--testbatchsize', default=8, type=int)

    args = vars(parse.parse_args())

    print(f"torch: {torch.__version__}, catalyst: {catalyst.__version__}, 'segmenttation pytorch version: {smp.__version__}")

    SEED = 42
    utils.set_global_seed(SEED)
    utils.prepare_cudnn(deterministic=True)

    MAIN = Path('../../')

    ROOT = MAIN / 'data/raw'

    train_image_dir = ROOT / '1. Original Images' / 'a. Training Set'
    train_mask_dir = ROOT / '2. All Segmentation Groundtruths' / 'a. Training Set'
    test_image_dir = ROOT / '1. Original Images' / 'b. Testing Set'
    test_mask_dir = ROOT / '2. All Segmentation Groundtruths' / 'b. Testing Set'

    CKPT = str(MAIN) + '/models/base_segmentation'
    OUTDIR = str(MAIN) + '/outputs'
    TTA_DIR = str(MAIN) + '/outputs/tta'

    for dir in [CKPT, OUTDIR, TTA_DIR]:
        if not os.path.isdir(dir):
            os.makedirs(dir, exist_ok=True)

    info = main(args)

    print('Normal test')
    test(args, info)

    print('TTA Test')
    test_tta(args, info)
