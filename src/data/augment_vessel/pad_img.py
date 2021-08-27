import os, cv2
import skimage.io as io
import skimage.transform as trans
import numpy as np
from pathlib import Path
import shutil

def pad(input_folder, output_folder, desired_size=608, already_padded=False, is_mask=False):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in sorted(os.listdir(input_folder)):
        print(file)
        if is_mask:
            tmp = io.imread(input_folder + '/' + file, as_gray=True)
            # tmp = cv2.cvtColor(tmp, cv2.COLOR_GR)
        else:
            tmp = cv2.imread(input_folder + '/' + file)

        if not already_padded:
            old_size = tmp.shape[:2]  # old_size is in (height, width) format
            delta_w = desired_size - old_size[1]
            delta_h = desired_size - old_size[0]

            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            if not is_mask:
                color = [0, 0, 0]
            else:
                color = [0]
            tmp = cv2.copyMakeBorder(tmp, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
            if is_mask:
                _, tmp = cv2.threshold(tmp, 127, 255, cv2.THRESH_BINARY)
                # print(tmp.shape)
        cv2.imwrite(output_folder + '/' + file, tmp)

    if not already_padded:
        print('Padding is done.')

def convert():
    raw_dir = Path('../../../data/processed/DRIVE_AUGMENT/raw/')

    if len(os.listdir(str(raw_dir /  'test/images'))) == 0:
        for file in sorted(os.listdir('../../../data/raw/DRIVE/test/HR')):
            shutil.copy('../../../data/raw/DRIVE/test/HR/' + file, '../../../data/processed/DRIVE_AUGMENT/raw/test/images/' + file.split('.')[0] + '.png')
    if len(os.listdir(str(raw_dir / 'test/labels'))) == 0:
        for file in sorted(os.listdir('../../../data/raw/DRIVE/test/VE')):
            shutil.copy('../../../data/raw/DRIVE/test/VE/' + file, '../../../data/processed/DRIVE_AUGMENT/raw/test/labels/' + file.split('.')[0] + '.png')


if __name__ == '__main__':
    # convert()
    # input_dir = '../../../data/processed/CHASE_AUGMENT/augment_1'
    input_dir = '../../../data/processed/CHASE_AUGMENT/raw/test'
    # input_test_dir = '../../../data/raw/DRIVE/test'
    des_dir = '../../../data/processed/CHASE_AUGMENT/pad_ver'

    # if not os.path.exists(des_dir):
    #     os.makedirs(des_dir)
    #     if not os.path.exists(des_dir + '/tmp_train'):
    #         os.makedirs(des_dir + '/tmp_train')
    #     if not os.path.exists(des_dir + '/tmp_test'):
    #         os.makedirs(des_dir + '/tmp_test')

    #pad imgs
    pad(input_dir + '/images', des_dir + '/test/images', desired_size=1024)
    #pad masks 
    pad(input_dir + '/labels', des_dir + '/test/labels', is_mask=True, desired_size=1024)
    
    # #pad test imgs
    # pad(input_test_dir + '/HR', des_dir + 'tmp_test/images')
    # #pad test masks
    # pad(input_test_dir + '/VE', des_dir + '/tmp_test/labels')
    pass