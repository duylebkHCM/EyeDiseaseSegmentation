from pathlib import Path
import os
from PIL import Image
import numpy as np
import cv2
from skimage import io
from matplotlib import image
from tqdm.auto import tqdm
import sys
sys.path.append('..')
from main.config import BaseConfig

if __name__ == '__main__':
    # img_path = '../..' / BaseConfig.train_img_path
    img_path = Path('../../data/processed/CHASEDB1/train/image')
    x_tot = []
    x2_tot = []
    for path in tqdm(list(img_path.glob('*.*'))):
        img = Image.open(path)
        # img= cv2.imread(str(path))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = io.imread(path)
        # img = image.imread(path)
        img = np.asarray(img).astype('float32')
        assert len(img.shape) == 3, \
            f'{len(img.shape)}-{img.max()}-{img.min()}-{path}'
        x_tot.append((img/255.0).reshape(-1, 3).mean(0))
        x2_tot.append(((img/255.0)**2).reshape(-1, 3).mean(0))

    img_avg = np.array(x_tot).mean(0)
    img_std = np.sqrt(np.array(x2_tot).mean(0) - img_avg**2)
    result = np.vstack([img_avg, img_std])
    print('mean:',img_avg, ', std:', img_std)
    np.savetxt(f'{BaseConfig.dataset_name}.txt', result, delimiter=',', fmt='%s')
