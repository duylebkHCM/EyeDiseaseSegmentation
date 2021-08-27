import os, shutil
from methods import *
from pathlib import Path
import glob

def apply(do_augment_path, ro30):
    if not os.path.exists(ro30):
        os.mkdir(ro30)
        rotation(30, do_augment_path, ro30, 20)
    else:
        print("This augmentation technique has been already done.")

def merge_augmentations(augment_dir, output_dir, list_of_aug_files):
    augment_dir = str(augment_dir)
    output_dir = str(output_dir)

    os.mkdir(output_dir + '/images')
    os.mkdir(output_dir + '/labels')

    for folder in list_of_aug_files:
        for file in sorted(os.listdir(augment_dir + '/' + folder + '/images')):
            shutil.copy(augment_dir + '/' + folder + '/images/' + file, output_dir + '/images/' + file.split("_")[-1].split('.')[0] + '.png')
        
            shutil.copy(augment_dir + '/' + folder + '/labels/' + file, output_dir + '/labels/' + file.split("_")[-1].split('.')[0] + '.png')

        print(folder + ' folder has been merged...')
        print('Number of images in output: ' + str(len(os.listdir(output_dir + '/images'))))
    print('Merging is done successfully!')


if __name__ == "__main__":
    main_dir = Path('../../../data/processed/CHASE_AUGMENT')
    raw_dir = Path('../../../data/processed/CHASE_AUGMENT/raw/')

    if len(os.listdir(str(raw_dir /  'train/images'))) == 0:
        for file in sorted(os.listdir('../../../data/raw/CHASE_DB1/training/images')):
            shutil.copy('../../../data/raw/DRIVE/training/images/' + file, '../../../data/processed/DRIVE_AUGMENT/raw/train/images/' + file.split('.')[0] + '.png')
    if len(os.listdir(str(raw_dir / 'train/labels'))) == 0:
        for file in sorted(os.listdir('../../../data/raw/CHASE_DB1/training/labels')):
            shutil.copy('../../../data/raw/DRIVE/training/labels/' + file, '../../../data/processed/DRIVE_AUGMENT/raw/train/labels/' + file.split('.')[0] + '.png')


    # augment_dir = main_dir /  "augmentation"

    # if not os.path.exists(augment_dir):
    #     os.mkdir(augment_dir)

    # apply('../../../data/raw/DRIVE/training', str(main_dir / 'augmentation/ro30'))

    # merge_augmentations_path = main_dir / "augmentation/augment_id_1"
    # if not os.path.exists(merge_augmentations_path):
    #   os.mkdir(merge_augmentations_path)

    # if not os.path.exists(str(augment_dir / "train")):
    #     os.mkdir(str(augment_dir / "train"))
    #     shutil.copytree(str(raw_dir/ "train/images"), str(augment_dir / "train/images/"))
    #     shutil.copytree(str(raw_dir / "train/labels"),str(augment_dir / "train/labels/"))

    # augment_list = ["train","ro30"]

    # merge_augmentations(augment_dir, merge_augmentations_path, augment_list)  



