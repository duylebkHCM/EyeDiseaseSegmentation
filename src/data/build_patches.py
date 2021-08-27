from typing import List
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import gc
import os
import shutil
import rasterio
import pandas as pd
from rasterio.windows import Window
from sklearn.model_selection import train_test_split
import sys
sys.path.append('..')
from main.util import make_grid, get_datapath


def build_patches(images: List[Path], mask_paths: List[Path] = None, out_imgs: Path=None, out_mask: Path = None):
    masks = []
    window = 256
    overlap = 32

    identity = rasterio.Affine(1, 0, 0, 0, 1, 0)
    neg_slide = 0
    pos_slide = 0
    neg_names = []

    for i, img_path in enumerate(tqdm(images)):
        with rasterio.open(img_path, transform = identity) as dataset:
            mask = Image.open(mask_paths[i]).convert('L')
            mask = mask.point(lambda x: 255 if x > 50 else 0, '1')
            mask = np.asarray(mask).astype('uint8')
            masks.append(mask)
            slices = make_grid(dataset.shape, window=window, min_overlap=overlap)
            
            for j, slc in enumerate(slices):
                x1,x2,y1,y2 = slc                        
                image = dataset.read([1,2,3],
                    window=Window.from_slices((x1,x2),(y1,y2)))
                image = np.moveaxis(image, 0, -1)
                if masks[-1][x1:x2,y1:y2].sum() > 0:
                    image = Image.fromarray(image).convert('RGB')
                    image_name = img_path.name[:-4] + '_patch_' + str(j) + '.jpg'
                    image.save(out_imgs /  image_name, quality=100, subsampling=0)
                    mask = masks[-1][x1:x2,y1:y2]
                    mask = Image.fromarray(mask*255)
                    mask.save(out_mask / image_name, quality=100, subsampling=0)
                    pos_slide +=1
                else:
                    image = Image.fromarray(image).convert('RGB')
                    image_name = img_path.name[:-4] + '_patch_' + str(j) + '.jpg'
                    image.save(out_imgs /  image_name, quality=100, subsampling=0)
                    mask = masks[-1][x1:x2,y1:y2]
                    mask = Image.fromarray(mask*255)
                    mask.save(out_mask / image_name, quality=100, subsampling=0)
                    neg_slide +=1
                    neg_names.append(image_name)

    del masks
    gc.collect()
    print('Number of pos slide', pos_slide)
    print('Number of neg slide', neg_slide)
    if neg_slide > pos_slide:
        rm_slide = neg_slide - pos_slide
        rm_idxs = np.random.choice(range(len(neg_names)), rm_slide, replace=False)
        neg_names = np.array(neg_names)
        rm_names = neg_names[rm_idxs].tolist()
        img_lists = os.listdir(out_imgs)
        mask_lists = os.listdir(out_mask)
        print('Before remove')
        print('-'*10)
        print('Number imgs', len(os.listdir(out_imgs)))
        print('Number mask', len(os.listdir(out_mask)))
        
        for img, mask in zip(img_lists, mask_lists):
            if img in rm_names:
                os.remove(out_imgs/img)
                os.remove(out_mask/mask)

        print('-'*10)
        print('After remove')
        print('Number imgs', len(os.listdir(out_imgs)))
        print('Number mask', len(os.listdir(out_mask)))

def build_dataframe(img_dirs: Path, mask_dirs: Path):
    img_names = sorted([img_dirs / f_name for f_name in os.listdir(img_dirs)])
    mask_names = sorted([mask_dirs / f_name for f_name in os.listdir(mask_dirs)])
    df = pd.DataFrame({'img': img_names, 'mask': mask_names})
    df.to_csv(img_dirs.parent / 'img_mask.csv', index = False, header=True)

if __name__ == '__main__':
    LESION_TYPE = 'Vessel_DRIVE'
    TRAIN_IMG_DIRS = Path('../../data/processed/DRIVE/train/image')
    TRAIN_MASK_DIRS = Path('../../data/processed/DRIVE/train/mask')

    images, masks = get_datapath(
            img_path=TRAIN_IMG_DIRS, mask_path=TRAIN_MASK_DIRS, lesion_type=LESION_TYPE)
    indices = np.arange(len(images))

    train_indices, valid_indices = train_test_split(
        indices, test_size=0.2, random_state=1999, shuffle=True)

    np_images = np.array(images)
    train_imgs = np_images[train_indices].tolist()
    valid_imgs = np_images[valid_indices].tolist()
    np_masks = np.array(masks)
    train_masks = np_masks[train_indices].tolist()
    valid_masks = np_masks[valid_indices].tolist()


    if LESION_TYPE.split('_')[1] == 'DRIVE':
        test_outimgs = Path('../../data/processed/DRIVE/test') / 'image'
        if not os.path.exists(test_outimgs):
            os.makedirs(str(test_outimgs))   
        test_outmasks = Path('../../data/processed/DRIVE/test') / 'mask'

        if not os.path.exists(test_outmasks):
            os.makedirs(str(test_outmasks))     
        print('Start copying...')
        for img_path in valid_imgs:
            img_name = img_path.name
            shutil.copyfile(img_path, test_outimgs / img_name)
        for mask_path in valid_masks:
            mask_name = mask_path.name
            shutil.copyfile(mask_path, test_outmasks / mask_name)
        print('Success')

    train_outimgs = Path('../../data/processed/DRIVE/train/') / 'image_patches'
    if not os.path.exists(train_outimgs):
        os.makedirs(str(train_outimgs))
    train_outmasks = Path('../../data/processed/DRIVE/train/') / 'mask_patches'
    if not os.path.exists(train_outmasks):
        os.makedirs(str(train_outmasks))

    val_outimgs = Path('../../data/processed/DRIVE/val/') / 'image_patches'
    if not os.path.exists(val_outimgs):
        os.makedirs(str(val_outimgs))    
    val_outmasks = Path('../../data/processed/DRIVE/val/') / 'mask_patches'
    if not os.path.exists(val_outmasks):
        os.makedirs(str(val_outmasks))


    #Save train img, mask
    build_patches(train_imgs, train_masks, train_outimgs, train_outmasks)
    #Save dataframe
    build_dataframe(train_outimgs, train_outmasks)
    
    #Save val img, mask
    build_patches(valid_imgs, valid_masks, val_outimgs, val_outmasks)
    #Save dataframe
    build_dataframe(val_outimgs, val_outmasks)