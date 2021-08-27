import os
from pathlib import Path
import pandas as pd
import numpy as np

if __name__ == '__main__':
    path = '../../data/raw/FGADR/Seg-set'

    img_dir = Path(path) / 'Original_Images'
    masks = {}
    for c, ds, _ in os.walk(str(self.dir)):
        for d in ds:
            d = os.path.join(c, d)
            for f in os.listdir(d):
                f_name = f[:8]
                img_name = img_name.split('.')[0][:8]
                if f_name == img_name:
                    mask_type = d.split('/')[-1]
                    if mask_type in list(lesion.keys()):
                        masks[lesion[mask_type]] = os.path.join(d, f)
                        break