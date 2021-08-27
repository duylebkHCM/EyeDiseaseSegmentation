
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import numpy as np
import random
import functools, os, time
import threading
import logging
import cv2
import albumentations as A
import imageio
import imgaug as ia
import imgaug.augmenters as iaa
import os 



logger = logging.getLogger(__name__)
ImageFile.LOAD_TRUNCATED_IMAGES = True


class DataAugmentation:

    def __init__(self):
        pass

    @staticmethod
    def openImage(image):
        return Image.open(image, mode="r")

    @staticmethod
    def randomRotation(image, label, mode=Image.BICUBIC):

        random_angle = np.random.randint(1, 360)
        return image.rotate(random_angle, mode), label.rotate(random_angle, Image.NEAREST)

    @staticmethod
    def randomColor(image, label):
        random_factor = np.random.randint(0, 31) / 10.  
        color_image = ImageEnhance.Color(image).enhance(random_factor)  
        random_factor = np.random.randint(10, 21) / 10.  
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  
        random_factor = np.random.randint(10, 21) / 10.  
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  
        random_factor = np.random.randint(0, 31) / 10.  
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor), label  

    @staticmethod
    def randomGaussian(image, label, mean=0.2, sigma=0.3):

        def gaussianNoisy(im, mean=0.2, sigma=0.3):

            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im

        img = np.array(image)
        img.flags.writeable = 1
        width, height = img.shape[:2]
        img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
        img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
        img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
        img[:, :, 0] = img_r.reshape([width, height])
        img[:, :, 1] = img_g.reshape([width, height])
        img[:, :, 2] = img_b.reshape([width, height])
        return Image.fromarray(np.uint8(img)), label
    
    # @staticmethod
    # def randomDropout(image, label, rate):
    @staticmethod
    def randomFlipHR(image, label):
        image = np.asarray(image)
        label = np.asarray(label)
        seq = iaa.Sequential(
            [
                iaa.Fliplr()
            ]
        )

        image_aug = seq(images=[image])
        label_aug = seq(images=[label])

        image = Image.fromarray(image_aug)
        label = Image.fromarray(label_aug)
        return image, label

    @staticmethod
    def randomFlipVT(image, label):
        image = np.asarray(image)
        label = np.asarray(label)
        seq = iaa.Sequential(
            [
                iaa.Flipud()
            ]
        )

        image_aug = seq(images=[image])
        label_aug = seq(images=[label])

        image = Image.fromarray(image_aug)
        label = Image.fromarray(label_aug)
        return image, label


    @staticmethod
    def saveImage(image, path):
        image.save(path)
        print('Saved to {}'.format(path))

def makeDir(path):
    try:
        if not os.path.exists(path):
            if not os.path.isfile(path):
                # os.mkdir(path)
                os.makedirs(path)
            return 0
        else:
            return 1
    except Exception:
        print(str(Exception))


def imageOps(func_name, image, label, img_des_path, label_des_path, img_file_name, label_file_name, times=3):
    funcMap = {"randomRotation": DataAugmentation.randomRotation,
               "randomColor": DataAugmentation.randomColor,
               "randomGaussian": DataAugmentation.randomGaussian,
               "randomFlipHR": DataAugmentation.randomFlipHR,
               "randomFlipVT": DataAugmentation.randomFlipVT
               }
    if funcMap.get(func_name) is None:
        logger.error("%s is not exist", func_name)
        return -1

    for _i in range(0, times, 1):
        new_image, new_label = funcMap[func_name](image, label)
        # print('Label', new_label.size)
        DataAugmentation.saveImage(new_image, os.path.join(img_des_path, func_name + str(_i) + img_file_name))
        DataAugmentation.saveImage(new_label, os.path.join(label_des_path, func_name + str(_i) + label_file_name))



def threadOPS(img_path, new_img_path, label_path, new_label_path, opsList):
    # if not os.path.exists(new_img_path):
    #     os.makedirs(new_img_path)
    #     if not os.path.exists(new_img_path + '/images'):
    #         os.makedirs(new_img_path + '/images')
    #     if not os.path.exists(new_img_path + '/labels'):
    #         os.makedirs(new_img_path + '/labels')

    # img path
    print(img_path)
    if os.path.isdir(img_path):
        img_names = os.listdir(img_path)
        print(img_names)
    else:
        img_names = [img_path]

    # label path
    if os.path.isdir(label_path):
        label_names = os.listdir(label_path)
        print(label_names)
    else:
        label_names = [label_path]

    img_num = 0
    label_num = 0

    # img num
    for img_name in img_names:
        tmp_img_name = os.path.join(img_path, img_name)
        if os.path.isdir(tmp_img_name):
            print('contain file folder')
            exit()
        else:
            img_num = img_num + 1;
    # label num
    for label_name in label_names:
        tmp_label_name = os.path.join(label_path, label_name)
        if os.path.isdir(tmp_label_name):
            print('contain file folder')
            exit()
        else:
            label_num = label_num + 1

    if img_num != label_num:
        print('the num of img and label is not equl')
        exit()
    else:
        num = img_num

    for i in range(num):
        # print('Hello')
        img_name = img_names[i]

        label_name = label_names[i]

        tmp_img_name = os.path.join(img_path, img_name)
        tmp_label_name = os.path.join(label_path, label_name)

        # print(tmp_img_name)
        image = DataAugmentation.openImage(tmp_img_name)
        label = DataAugmentation.openImage(tmp_label_name)

        threadImage = [0] * 5
        _index = 0
        for ops_name in opsList:
            threadImage[_index] = threading.Thread(target=imageOps,
                                                   args=(ops_name, image, label, new_img_path, new_label_path, img_name,
                                                         label_name))
            threadImage[_index].start()
            _index += 1
            time.sleep(5)

# Please modify the path
if __name__ == '__main__':
    opsList = {"randomRotation", "randomGaussian", "randomColor"}
    threadOPS("../../../data/processed/CHASE_AUGMENT/raw/train/images", #set your path of training images
              "../../../data/processed/CHASE_AUGMENT/augment_1/images/",
              "../../../data/processed/CHASE_AUGMENT/raw/train/labels",# set your path of training labels
              "../../../data/processed/CHASE_AUGMENT/augment_1/labels",
              opsList
            )