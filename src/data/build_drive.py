import os
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm 

if __name__ == "__main__":
    des_dir = Path('../../data/processed/DRIVE/')
    source_dir = Path('../../data/raw/DRIVE/')
    
    # if not os.path.exists(des_dir / 'train' /'image'):
    #     os.makedirs(des_dir / 'train' /'image')
    # if not os.path.exists(des_dir / 'train' /'mask'):
    #     os.makedirs(des_dir / 'train' /'mask')
    if not os.path.exists(des_dir / 'test' /'image'):
        os.makedirs(des_dir / 'test' /'image')
    if not os.path.exists(des_dir / 'test' /'mask'):
        os.makedirs(des_dir / 'test' /'mask')

    # for i, img in enumerate(tqdm(sorted((source_dir / 'training' / 'images').glob('*.tif')))):
    #     src_img = Image.open(img)
    #     src_img.save(des_dir / 'train' /'image' / (str(i) + '.jpg'), quality=100, subsampling=0)

    # for i, img in enumerate(tqdm(sorted((source_dir / 'training' / '1st_manual').glob('*.gif')))):
    #     src_img = Image.open(img)
    #     src_img.save(des_dir / 'train' /'mask' /(str(i) + '.jpg'), quality=100, subsampling=0)

    # for i, img in enumerate(tqdm(sorted((source_dir / 'test' / 'HR').glob('*.tif')))):
    #     src_img = Image.open(img)
    #     src_img.save(des_dir / 'test' /'image' /(str(i) + '.jpg'), quality=100, subsampling=0)

    for i, img in enumerate(tqdm(sorted((source_dir / 'test' / 'VE').glob('*.gif')))):
        src_img = Image.open(img)
        src_img.save(des_dir / 'test' /'mask' /(str(i) + '.jpg'), quality=100, subsampling=0)