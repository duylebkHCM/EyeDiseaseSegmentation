#!/usr/bin/env python
# -*- coding: utf-8 -*-
# **************************************
# @Author  : Qiqi Xiao
# @Email     : xiaoqiqi177@gmail.com
# @File    : preprocess.py
# **************************************
#Inspired from: https://github.com/MasazI/clahe_python_opencv/blob/master/core.py

import sys
import glob
import os
import os.path
import cv2
import numpy as np

def clahe_gridsize(image_path, mask_path, denoise=False, brightnessbalance=None, cliplimit=None, gridsize=8):
    """This function applies CLAHE to normal RGB images and outputs them.
    The image is first converted to LAB format and then CLAHE is applied only to the L channel.
    Inputs:
      image_path: Absolute path to the image file.
      mask_path: Absolute path to the mask file.
      denoise: Toggle to denoise the image or not. Denoising is done after applying CLAHE.
      cliplimit: The pixel (high contrast) limit applied to CLAHE processing. Read more here: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html
      gridsize: Grid/block size the image is divided into for histogram equalization.
    Returns:
      bgr: The CLAHE applied image.
    """
    bgr = cv2.imread(image_path)
    
    # brightness balance.
    if brightnessbalance:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        mask_img = cv2.imread(mask_path, 0)
        brightness = gray.sum() / (mask_img.shape[0] * mask_img.shape[1] - mask_img.sum()/255.)
        bgr = np.uint8(np.minimum(bgr * brightnessbalance / brightness, 255))

    # illumination correction and contrast enhancement.
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=cliplimit,tileGridSize=(gridsize,gridsize))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    if denoise:
        bgr = cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 1, 3)
        bgr = cv2.bilateralFilter(bgr, 5, 1, 1)

    return bgr

def prepare_img(image_dir, preprocess=False, phase='train'):
    if phase == 'train':
        setname = 'a. Training Set'
    elif phase == 'test':
        setname = 'b. Testing Set' 
    if preprocess:
        limit = 2
        grid_size = 8
        if not os.path.exists(os.path.join(image_dir, 'Images_CLAHE')):
            os.mkdir(os.path.join(image_dir, 'Images_CLAHE'))
        if not os.path.exists(os.path.join(image_dir, 'Images_CLAHE', setname)):
            os.mkdir(os.path.join(image_dir, 'Images_CLAHE', setname))
            
            # compute mean brightess
            meanbright = 0.
            images_number = 0
            for tempsetname in ['a. Training Set', 'b. Testing Set']:
                imgs_ori = glob.glob(os.path.join(image_dir, '1. Original Images/'+tempsetname+'/*.jpg'))
                imgs_ori.sort()
                images_number += len(imgs_ori)
                # mean brightness.
                for img_path in imgs_ori:
                    img_name = os.path.split(img_path)[-1].split('.')[0]
                    mask_path = os.path.join(image_dir, '2. All Segmentation Groundtruths', tempsetname, 'Mask', img_name+'_MASK.tif')
                    gray = cv2.imread(img_path, 0)
                    mask_img = cv2.imread(mask_path, 0)
                    brightness = gray.sum() / (mask_img.shape[0] * mask_img.shape[1] - mask_img.sum() / 255.)
                    meanbright += brightness
            meanbright /= images_number
            
            imgs_ori = glob.glob(os.path.join(image_dir, '1. Original Images/'+setname+'/*.jpg'))
            
            for img_path in imgs_ori:
                img_name = os.path.split(img_path)[-1].split('.')[0]
                mask_path = os.path.join(image_dir, '2. All Segmentation Groundtruths', setname, 'Mask', img_name+'_MASK.tif')
                clahe_img = clahe_gridsize(img_path, mask_path, denoise=True, brightnessbalance=meanbright, cliplimit=limit, gridsize=grid_size)
                cv2.imwrite(os.path.join(image_dir, 'Images_CLAHE', setname, os.path.split(img_path)[-1]), clahe_img)

if __name__ == '__main__':
  prepare_img('../../data/raw/IDRiD/', True, phase='train')
  prepare_img('../../data/raw/IDRiD/', True, phase='test')
