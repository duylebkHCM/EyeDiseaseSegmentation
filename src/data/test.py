from PIL import Image
import numpy as np
import torch
from pytorch_toolbelt.utils.catalyst import (
    draw_binary_segmentation_predictions,
)
import sys
sys.path.append('..')
from main import archs
from data_transform import load_ben_color, NormalTransform, MediumTransform, AdvancedTransform
import matplotlib.pyplot as plt

def to_numpy(x:torch.Tensor) -> np.ndarray:
    """
    Convert whatever to numpy array
    Args:
        :param x: List, tuple, PyTorch tensor or numpy array
    Returns:
        :return: Numpy array
    """
    if torch.is_tensor(x):
        return x.data.cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, (list, tuple, int, float)):
        return np.array(x)
    else:
        raise ValueError("Unsupported type")

def rgb_image_from_tensor(
    image: torch.Tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), min_pixel_value=0.0, max_pixel_value=255.0, dtype=np.uint8
) -> np.ndarray:
    """
    Convert numpy image (RGB, BGR, Grayscale, SAR, Mask image, etc.) to tensor
    Args:
        image: A torch tensor of [C,H,W] shape
    """
    image = np.moveaxis(to_numpy(image), 0, -1)
    mean = to_numpy(mean)
    std = to_numpy(std)
    rgb_image = (max_pixel_value * (image * std + mean))
    rgb_image = np.clip(rgb_image, a_min=min_pixel_value, a_max=max_pixel_value)
    return rgb_image.astype(dtype)

if __name__ == '__main__':
    path = '../../data/processed/DRIVE/train/image/13.jpg'
    image = Image.open(path).convert('RGB')
    origin_image = np.asarray(image).astype('uint8')

    preprocessing, mean, std = archs.get_preprocessing_fn('DRIVE')
    
    transform = AdvancedTransform(512, preprocessing)
    preprocessing_fn = transform.get_preprocessing()

    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        image = load_ben_color(origin_image, img_size=(origin_image.shape[1], origin_image.shape[0]))
        image = transform.train_transform()(image=image)['image']
        image = preprocessing_fn(image=image)['image']

        image = torch.FloatTensor(image).permute(2, 0, 1)
        image = rgb_image_from_tensor(image, mean=mean, std=std)
        image = Image.fromarray(image)
        ax.imshow(image)    
    
    plt.savefig('test.jpg')   

