#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Author  : Qiqi Xiao
# @Email     : xiaoqiqi177@gmail.com
# @File    : dataset.py
# **************************************
import numpy as np
from torchvision import datasets, models, transforms
import torchvision
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class IDRIDDataset(Dataset):

    def __init__(self, image_paths, mask_paths=None, class_id=0, transform=None):
        """
        Args:
            image_paths: paths to the original images []
            mask_paths: paths to the mask images, [[]]
            class_id: id of lesions, 0:ex, 1:he, 2:ma, 3:se
        """
        assert len(image_paths) == len(mask_paths)
        self.image_paths = []
        self.mask_paths = []
        self.masks = []
        self.images = []
        if self.mask_paths is not None:
            for image_path, mask_path4 in zip(image_paths, mask_paths):
                mask_path = mask_path4[class_id]
                if mask_path is None:
                    continue
                else:
                    self.image_paths.append(image_path)
                    self.mask_paths.append(mask_path)
                    self.images.append(self.pil_loader(image_path))
                    self.masks.append(self.pil_loader(mask_path))
        
        self.class_id = class_id
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def pil_loader(self, image_path):
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            h, w = img.size
            #return img.resize((h//2, w//2)).convert('RGB')
            return img.convert('RGB')

    def __getitem__(self, idx):
        info = [self.images[idx]]
        if self.mask_paths:
            info.append(self.masks[idx])
        if self.transform:
            info = self.transform(info)

        inputs = np.array(info[0])
        if inputs.shape[2] == 3:
            inputs = np.transpose(np.array(info[0]), (2, 0, 1))
            inputs = inputs / 255.
        
        if len(info) > 1:
            mask = np.array(np.array(info[1]))[:, :, 0] / 255.0
            empty_mask = 1 - mask
            masks = np.array([empty_mask, mask])

            return inputs, masks
        else:
            return inputs

import os
def get_paths(imgs, phase, ratio):
    imgs.sort()     
    mask_paths = []
    train_number = int(len(imgs) * ratio)
    if phase == 'train':
        image_paths = imgs[:train_number]
    elif phase == 'eval':
        image_paths = imgs[train_number:]
    else:
        image_paths = imgs
    mask_path = os.path.join('../../data/raw/IDRiD', '2. All Segmentation Groundtruths', 'a. Training Set')
    lesions = ['3. Hard Exudates', '2. Haemorrhages', '1. Microaneurysms', '4. Soft Exudates', 'Mask']
    lesion_abbvs = ['EX', 'HE', 'MA', 'SE', 'MASK']
    for image_path in image_paths:
        paths = []
        name = os.path.split(image_path)[1].split('.')[0]
        for lesion, lesion_abbv in zip(lesions, lesion_abbvs):
            candidate_path = os.path.join(mask_path, lesion, name+'_'+lesion_abbv+'.tif')
            if os.path.exists(candidate_path):
                paths.append(candidate_path)
            else:
                paths.append(None)
        mask_paths.append(paths)
    return image_paths, mask_paths


def pil_loader(image_path):
    with open(image_path, 'rb') as f:
        img = Image.open(f)
        h, w = img.size
        #return img.resize((h//2, w//2)).convert('RGB')
        return img.convert('RGB')

if __name__ == '__main__':
    import glob
    imgs = glob.glob(os.path.join('../../data/raw/IDRiD', 'Images_CLAHE', 'a. Training Set', '*.jpg'))
    train_paths, train_mask_paths = get_paths(imgs, 'train', 0.8)
    # print('Train paths', train_paths)
    # print('Maks path', train_mask_paths)

    
    import torchvision.transforms as transforms
    transform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomCrop(512),
                ])

    # dataset = IDRIDDataset(train_paths, train_mask_paths, 0)
    # loader = DataLoader(dataset, 2)

    # batch = next(iter(loader))
    # for img, mask in batch:
    #     print('Image', img.shape)
    #     print('Maks', mask.shape)
    
    img = train_paths[0]
    mask = train_mask_paths[0][0]

    img = pil_loader(img)
    mask = pil_loader(mask)

    np_mask = np.asarray(mask).astype('uint8')
    print(np_mask.shape)
    print('Unique', np.unique(np_mask))
    info = mask

    mask = transform(info)

    mask = np.array(np.array(mask))[:, :, 0] / 255.0
    empty_mask = 1 - mask
    masks = np.array([empty_mask, mask])
    # print(type(out[0]))
    # print(type(out[1]
    # ))


    print(masks.shape)