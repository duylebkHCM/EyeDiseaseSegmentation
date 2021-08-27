import os
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm 
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    des_dir = Path('../../data/processed/HRF/')
    source_dir = Path('../../data/raw/HRF/')
    
    if not os.path.exists(des_dir / 'train' /'image'):
        os.makedirs(des_dir / 'train' /'image')
    if not os.path.exists(des_dir / 'train' /'mask'):
        os.makedirs(des_dir / 'train' /'mask')
    if not os.path.exists(des_dir / 'test' /'image'):
        os.makedirs(des_dir / 'test' /'image')
    if not os.path.exists(des_dir / 'test' /'mask'):
        os.makedirs(des_dir / 'test' /'mask')

    split_propotion = 0.2 
    train_img, test_img = train_test_split(list((source_dir/'images').glob('*.*')), test_size=split_propotion, shuffle=True, random_state=1999)

    train_img = sorted(train_img)
    test_img = sorted(test_img)

    for i, img in enumerate(tqdm(train_img)):
        src_img = Image.open(img)
        src_img.save(des_dir / 'train' /'image' / (str(i) + '.jpg'), quality=100, subsampling=0)
        mask = img.name[:-4] + '.tif'
        src_mask = Image.open(source_dir / 'manual1' / mask)
        src_mask.save(des_dir/'train'/'mask'/(str(i) + '.jpg'), quality=100, subsampling=0)

    for i, img in enumerate(tqdm(test_img)):
        src_img = Image.open(img)
        src_img.save(des_dir / 'test' /'image' / (str(i) + '.jpg'), quality=100, subsampling=0)
        mask = img.name[:-4] + '.tif'
        src_mask = Image.open(source_dir / 'manual1' / mask)
        src_mask.save(des_dir/'test'/'mask'/(str(i) + '.jpg'), quality=100, subsampling=0)

