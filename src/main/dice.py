import torch
import segmentation_models_pytorch as smp
from PIL import Image
from pathlib import Path
import ttach as tta
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
import cv2
import torch.cuda.amp as amp
import rasterio
from rasterio.windows import Window

import sys
sys.path.append('..')
from config import TestConfig
import archs
import util
from data import TestSegmentation, NormalTransform
from util import make_grid

def get_model(params, model_name):
    # Model return logit values
    params['encoder_weights'] = None
    model = getattr(smp, model_name)(
        **params
    )
    return model

if __name__ == '__main__':
    exp_name = 'Apr26_09_24'
    config = TestConfig.get_all_attributes()
    logdir = '../../models/' + config['dataset_name'] + '/' + config['lesion_type'] + '/' + exp_name

    if hasattr(smp, config['model_name']):
        model = get_model(
            config['model_params'], config['model_name'])
    elif config['model_name'] == "TransUnet":
        from self_attention_cv.transunet import TransUnet
        model = TransUnet(**config['model_params'])
    else:
        model = archs.get_model(
            model_name=config['model_name'], 
            params = config['model_params'], training=False)
    preprocessing_fn, mean, std = archs.get_preprocessing_fn(dataset_name=config['dataset_name'])

    test_img_dir = '../..' / config['test_img_path']
    test_mask_dir = '../..' / config['test_mask_path']

    img_dirs, mask_dirs = util.get_datapath(
            img_path=test_img_dir, mask_path=test_mask_dir, lesion_type=config['lesion_type'])

    transform = NormalTransform(config['scale_size'])
    test_transform = A.Compose([
            A.Resize(256, 256),
            A.Lambda(image = preprocessing_fn),
            ToTensorV2()
        ])

    test_transform1 = A.Compose(transform.resize_transforms() + [A.Lambda(image = preprocessing_fn),ToTensorV2()])  
    
    if config['data_type'] == 'tile':
        test_ds = TestSegmentation(img_dirs, mask_dirs)
    else:
        test_ds = TestSegmentation(img_dirs, mask_dirs, test_transform1)
        test_loader = DataLoader(test_ds, batch_size=1, num_workers=2, pin_memory=True, shuffle=True)

    checkpoints = torch.load(f"{logdir}/checkpoints/best.pth")
    model.load_state_dict(checkpoints['model_state_dict'])
    model.eval()
    model = model.to('cuda')
    model = tta.SegmentationTTAWrapper(
            model, tta.aliases.d4_transform(), merge_mode="mean")

    mean_dice = 0
    mean_iou = 0
    mean_auc = 0
    mean_pre = 0
    mean_re = 0
    dice_score = smp.utils.metrics.Fscore(threshold=None)
    iou_score = smp.utils.metrics.IoU(threshold=None)
    precision = smp.utils.metrics.Precision(threshold=None)
    recall = smp.utils.metrics.Recall(threshold=None)

    if config['data_type'] == 'tile':
        with torch.no_grad():
            for item in tqdm(test_ds, total=len(test_ds)):
                image_name = item['filename']
                img = test_img_dir / image_name
                gt_mask = torch.from_numpy(item['mask'])
                with rasterio.open(img.as_posix(), transform=rasterio.Affine(1, 0, 0, 0, 1, 0)) as dataset:
                    slices = make_grid(dataset.shape, window=512, min_overlap=32)
                    preds = np.zeros(dataset.shape, dtype=np.float32)

                    for (x1, x2, y1, y2) in slices:
                        image = dataset.read([1,2,3], window = Window.from_slices((x1, x2), (y1, y2)))
                        image = np.moveaxis(image, 0, -1)
                        image = test_transform(image=image)['image']
                        image = image.float()
                        
                        with torch.no_grad():
                            image = image.to('cuda')[None]
                            logit = model(image)[0][0]
                            score_sigmoid = logit.sigmoid().cpu().numpy()
                            score_sigmoid = cv2.resize(score_sigmoid, (512, 512), interpolation=cv2.INTER_LINEAR)
                            preds[x1:x2, y1:y2] = score_sigmoid
                    preds = torch.from_numpy(preds)

                dice = dice_score(preds, gt_mask)
                iou = iou_score(preds, gt_mask)
                pre = precision(preds, gt_mask)
                rec = recall(preds, gt_mask)
                auc_score = average_precision_score(gt_mask.numpy().reshape(-1), preds.numpy().reshape(-1))
                mean_dice += dice.item()
                mean_iou += iou.item()  
                mean_auc += auc_score.item()
                mean_pre += pre.item()
                mean_re += rec.item()

        print('DICE',mean_dice / len(test_ds))
        print('IOU',mean_iou / len(test_ds))
        print('MEAN-AUC', mean_auc / len(test_ds))
        print('PRECISION', mean_pre / len(test_ds))
        print('RECALL', mean_re / len(test_ds))
        
    else:
        with torch.no_grad():
            for batch in test_loader:
                pred = model(batch['image'].to('cuda'))
                pred = pred.detach().cpu()
                pred = torch.sigmoid(pred)

                iou = iou_score(pred, batch['mask'])
                dice = dice_score(pred, batch['mask'])
                pre = precision(pred, batch['mask'])
                rec = recall(pred, batch['mask'])
                auc_score = average_precision_score(batch['mask'].numpy().reshape(-1), pred.numpy().reshape(-1))
                mean_iou += iou.item()
                mean_dice += dice.item()
                mean_auc += auc_score.item()
                mean_pre += pre.item()
                mean_re += rec.item()
                print(dice.item(), batch['filename'][0])

        print('IOU', mean_iou/len(test_loader))
        print('DICE', mean_dice/len(test_loader))
        print('AUC-PR', mean_auc/len(test_loader))
        print('PPV', mean_pre/len(test_loader))
        print('SN', mean_re/len(test_loader))