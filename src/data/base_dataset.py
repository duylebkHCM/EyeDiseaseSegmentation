from torch.utils.data import Dataset
from typing import List
from catalyst.contrib.utils.cv import image as cata_image
from pathlib import Path
from skimage.io import imread as mask_read
import numpy as np


class LesionSegmentation(Dataset):
    def __init__(self, images: List[Path], masks: List[Path]=None, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
        image_path = self.images[index]

        image = cata_image.imread(image_path)
        results = {'image': image}

        if self.masks is not None:
            mask = mask_read(self.masks[index]).astype(np.float32)
            results['mask'] = mask

        if self.transform is not None:
            results = self.transform(**results)

        results['filename'] = image_path.name

        return results
