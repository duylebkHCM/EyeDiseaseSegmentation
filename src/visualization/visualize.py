import random
import os
import cv2
import re
from tqdm.auto import tqdm
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Callable
from PIL import Image
from skimage.io import imread as mask_read
from catalyst.contrib.utils.cv import image as cata_image

def show_examples(name: str, image: np.ndarray, mask: np.ndarray):
    plt.figure(figsize=(10, 14))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Image: {name}")

    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title(f"Mask: {name}")


def show(index: int, images: List[Path], masks: List[Path], transforms=None) -> None:
    image_path = images[index]
    name = image_path.name

    image = cata_image.imread(image_path)
    mask = mask_read(masks[index]).astype(np.float32)

    if transforms is not None:
        temp = transforms(image=image, mask=mask)
        image = temp["image"]
        mask = temp["mask"]

    show_examples(name, image, mask)


def show_random(images: List[Path], masks: List[Path], transforms=None) -> None:
    length = len(images)
    index = random.randint(0, length - 1)
    show(index, images, masks, transforms)

def overlay_image_mask(image, mask, mask_color=(0,255,0), alpha=1.0):
    im_f= image.astype(np.float32)
#     if mask.ndim == 2:
#         mask = np.expand_dims(mask,-1)        
    mask_col = np.expand_dims(np.array(mask_color)/255.0, axis=(0,1))
    return (im_f + alpha * mask * (np.mean(0.8 * im_f + 0.2 * 255, axis=2, keepdims=True) * mask_col - im_f)).astype(np.uint8)


def overlay_image_mask_original(image, mask, mask_color=(0,255,0), alpha=1.0):
    return  np.concatenate((image, overlay_image_mask(image, mask)), axis=1)

def overlay_mask_image(
    image: Path, 
    groundtruth: Path,
    binary_mask: Path,
    lesion_type: str,
    is_save: bool
) -> List[np.ndarray]:
    """
    Render visualization of model's prediction for binary segmentation problem.
    This function draws a color-coded overlay on top of the image, with color codes meaning:
        - green: True positives
        - red: False-negatives
        - yellow: False-positives
    """
    exp_name = binary_mask.name
    if is_save:
        save_dir = binary_mask.parent.parent.parent / 'gt_vs_prd' / lesion_type / exp_name
        if not os.path.exists(save_dir):    
            os.makedirs(save_dir, exist_ok=True)

    gt_paths = list(groundtruth.glob('*.tif'))
    for i in tqdm(range(len(gt_paths))):
        img_name = re.sub('_' + lesion_type + '.tif', '.jpg', gt_paths[i].name)
        img = cv2.imread(str(image/img_name), cv2.IMREAD_COLOR)
        overlay = img.copy()
        true_mask = cv2.imread(str(gt_paths[i]), cv2.IMREAD_GRAYSCALE) / true_mask.max()
        true_mask = true_mask.astype(np.uint8)
        pred_mask = cv2.imread(str(binary_mask / img_name), cv2.IMREAD_GRAYSCALE)
        _, pred_mask = cv2.threshold(pred_mask, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        pred_mask = (pred_mask / 255).astype(np.uint8)

        # overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        print(overlay.dtype)
        overlay[true_mask & pred_mask] = np.array(
            [0, 250, 0], dtype=overlay.dtype
        )  # Correct predictions (Hits) painted with green
        # overlay[true_mask & ~pred_mask] = np.array([250, 0, 0], dtype=overlay.dtype)  # Misses painted with red
        # overlay[~true_mask & pred_mask] = np.array(
        #     [250, 250, 0], dtype=overlay.dtype
        # )  # False alarm painted with yellow
        overlay = cv2.addWeighted(img, 0.5, overlay, 0.5, 0, dtype=cv2.CV_8U)
        cv2.putText(overlay, str(img_name[:-4]), (10, 15), cv2.FONT_HERSHEY_PLAIN, 10, (250, 250, 250))
        
        if is_save:
            cv2.imwrite(str(save_dir / img_name), overlay)
            print('Saved!', img_name)

if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from main.config import TestConfig
    # from main.util.base_utils import lesion_dict
    # overlay_mask_image(
    #     '../..' / TestConfig.test_img_path, 
    #     '../..'/ TestConfig.test_mask_path / lesion_dict['SE'].dir_name, 
    #     Path('../../outputs/IDRiD/tta/SE/Apr21_18_47'),
    #     lesion_type='SE',
    #     is_save=True)

    img = cv2.imread(str('../..' / TestConfig.test_img_path/'IDRiD_55.jpg'), cv2.IMREAD_COLOR)
    true_mask = cv2.imread(str('../../data/raw/IDRiD/2. All Segmentation Groundtruths/b. Testing Set/3. Hard Exudates/IDRiD_55_EX.tif'), cv2.IMREAD_COLOR) 
    # true_mask = true_mask / true_mask.max()
    # true_mask = true_mask.astype(np.uint8)

    overlay_img = overlay_image_mask(img, true_mask)
    cv2.imwrite('test.jpg', overlay_img)