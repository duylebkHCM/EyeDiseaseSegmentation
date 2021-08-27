from pathlib import Path
import os
import numpy as np
import re
import cv2
from skimage.io import imread as mask_read
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == "__main__":

    TRAIN_IMG_DIRS = Path('../../data/raw/IDRiD/1. Original Images/a. Training Set')
    TRAIN_MASK_DIRS = Path(
        '../../data/raw/IDRiD/2. All Segmentation Groundtruths/a. Training Set')

    TEST_IMG_DIRS = Path('../../data/raw/IDRiD/1. Original Images/b. Testing Set')
    TEST_MASK_DIRS = Path('../../data/raw/IDRiD/2. All Segmentation Groundtruths/b. Testing Set')


    CLASS_COLORS = [1, 10, 20, 30]

    CLASS_NAMES = [
        'MA',
        'EX',
        'HE',
        'SE'
    ]

    lesion_paths = {
        'MA': '1. Microaneurysms',
        'EX': '3. Hard Exudates',
        'HE': '2. Haemorrhages',
        'SE': '4. Soft Exudates'
    }

    save_path = Path('../../data/processed/multilesion')

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for image_path in list(TRAIN_IMG_DIRS.glob('*.jpg')):
        final_mask = np.zeros(np.asarray(Image.open(image_path)).shape[:2]).transpose(0, 1).astype(np.uint8)
        
        i = 0
        masks = []
        for clss in CLASS_NAMES:
            mask_name = re.sub('.jpg', '_' + clss + '.tif', image_path.name)
            path = os.path.join(TRAIN_MASK_DIRS, lesion_paths[clss], mask_name)
            if os.path.exists(path):
                mask = mask_read(path).astype(np.uint8)
                _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
                mask = (mask / 255).astype(np.uint8)

                assert list(np.unique(mask)) == [0, 1]

                final_mask = cv2.bitwise_or(final_mask, mask, dst=final_mask)
                final_mask = final_mask*CLASS_COLORS[i]

                # masks.append(mask)
                i += 1
        
        # final_mask = cv2.bitwise_or(final_mask, mask, dst=final_mask)
        print(np.unique(final_mask))
        final_mask = Image.fromarray(final_mask)
        final_mask.save(save_path / re.sub('.jpg', '.tif', image_path.name))
        # plt.savefig(sav)


