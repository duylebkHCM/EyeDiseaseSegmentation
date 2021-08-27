import torch
from torch.utils.data import Dataset
from typing import List
from pathlib import Path
from pytorch_toolbelt.utils import fs, image_to_tensor
import numpy as np
from iglovikov_helper_functions.utils.image_utils import pad
from collections import OrderedDict
from PIL import Image
import cv2
import albumentations.augmentations.geometric.functional as F

__all__ = ['CLASS_NAMES', 'CLASS_COLORS', 'OneLesionSegmentation', 'TestSegmentation']


CLASS_NAMES = [
    'MA',
    'EX',
    'HE',
    'SE',
]

CLASS_COLORS = [
    (192, 192, 128),
    (128, 64, 128),
    (0, 0, 192),
    (128, 128, 0)
]

lesion_paths = {
    'MA': '1. Microaneurysms',
    'EX': '3. Hard Exudates',
    'HE': '2. Haemorrhages',
    'SE': '4. Soft Exudates',
}

class OneLesionSegmentation(Dataset):
    def __init__(self, images: List[Path], is_gray: bool, masks: List[Path] = None, transform=None, preprocessing_fn=None, ben_transform = None, data_type = 'all'):
        self.images = images
        self.is_gray = is_gray
        self.mask_paths = masks
        self.transform = transform
        self.ben_transform = ben_transform
        self.preprocessing_fn = preprocessing_fn
        self.mode = data_type
        self.len = len(images)

    def __len__(self):
        return self.len

    def __getitem__(self, index: int) -> dict:
        if self.mode == 'all':
            image_path = self.images[index]
            image = Image.open(image_path).convert('RGB')
            image = np.asarray(image).astype(np.uint8)
            mask = Image.open(self.mask_paths[index]).convert('L')
            mask = mask.point(lambda x: 255 if x > 50 else 0, '1')
            mask = np.asarray(mask).astype(np.float32)
            image_id = fs.id_from_fname(image_path)
        else:
            image_path = self.images[index]
            image = Image.open(image_path).convert('RGB')
            image = np.asarray(image).astype(np.uint8)
            mask = Image.open(self.mask_paths[index]).convert('L')
            mask = mask.point(lambda x: 255 if x > 50 else 0, '1')
            mask = np.asarray(mask).astype(np.float32)
            image_id = fs.id_from_fname(image_path)
            
        if self.is_gray:
            image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype('uint8')

        if self.ben_transform is not None:
            image, mask = self.ben_transform(image, mask, img_size=(image.shape[1], image.shape[0]))

        if self.transform is not None:
            results = self.transform(image=image, mask=mask)
            image, mask = results['image'], results['mask']
        
        if self.preprocessing_fn is not None:
            result = self.preprocessing_fn(image=image)
            image = result['image']

        image = image_to_tensor(image).float()
        mask = image_to_tensor(mask, dummy_channels_dim=True).float()

        assert mask.shape[:1] == torch.Size([1]) and len(mask.shape) == 3, f'Mask shape is {mask.shape}'
        return {
            'image': image,
            'mask': mask,
            'image_id': image_id
        }

# VesselSegmentation = OneLesionSegmentation

class TestSegmentation(Dataset):
    def __init__(self, images: List[Path], is_gray: bool, masks: List[Path] = None, transform=None, factor=None):
        self.images = images
        self.masks = masks
        self.transform = transform
        self.factor = factor
        self.is_gray = is_gray
        image = Image.open(images[0]).convert('RGB')
        self.ori_w, self.ori_h = image.size
        image = np.asarray(image).astype('uint8')
        tmp_img = F.longest_max_size(image, 1024, cv2.INTER_LINEAR)
        self.crop_h, self.crop_w = tmp_img.shape[0], tmp_img.shape[1]
        del tmp_img

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
        image_path = self.images[index]

        result = OrderedDict()
        image = Image.open(image_path).convert('RGB')
        image = np.asarray(image).astype('uint8')

        if self.is_gray:
            image = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype('uint8')
            image = np.expand_dims(image, -1)

        result['image'] = image

        if self.masks is not None:  
            mask = Image.open(self.masks[index]).convert('L')
            mask = mask.point(lambda x: 255 if x > 50 else 0, '1')
            mask = np.asarray(mask).astype(np.uint8)
            result['mask'] = mask

        if self.transform is not None:
            transformed = self.transform(**result)
            image = transformed['image']
            image = image.float()
            result['image'] = image
            if self.masks is not None:
                result['mask'] = transformed['mask']

        # print('Mask shape', result['mask'].shape)
        result['filename'] = image_path.name
        if self.factor is not None:
            normalized_image, pads = pad(image, factor=self.factor)
            result['pad'] = np.array(pads)
            result['image'] = normalized_image

        return result