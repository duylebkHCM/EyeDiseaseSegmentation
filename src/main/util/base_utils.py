import re
from pathlib import Path
import collections
import os
import numpy as np
from typing import List, Callable, Union, Tuple
from PIL import Image
from catalyst.utils.distributed import (
    get_distributed_env,
    get_distributed_params,
)
import warnings
import subprocess
import sys
import torch

from prettytable import PrettyTable


Lesion = collections.namedtuple('Lesion', ['dir_name', 'project_name'])

lesion_dict = {
    'MA': Lesion(dir_name='1. Microaneurysms', project_name='MicroaneurysmsSegmentation'),
    'EX': Lesion(dir_name='3. Hard Exudates', project_name='HardExudatesSegmentation'),
    'HE': Lesion(dir_name='2. Haemorrhages',
           project_name='HaemorrhageSegmentation'),
    'SE': Lesion(dir_name='4. Soft Exudates',
           project_name='SoftExudatesSegmentation'),
    'MA_DDR': Lesion(dir_name='MA', project_name='DDRMicroaneurysmsSegmentation'),
    'EX_DDR': Lesion(dir_name='EX', project_name='DDRHardExudatesSegmentation'),
    'HE_DDR': Lesion(dir_name='HE', project_name='DDRHaemorrhageSegmentation'),
    'SE_DDR': Lesion(dir_name='SE',project_name='DDRSoftExudatesSegmentation'),
    'OD': Lesion(dir_name='5. Optic Disc', project_name='OpticDiscSegmentation'),
    'EX_FGADR': Lesion(dir_name='HardExudate_Masks', project_name='FGADRHardExudatesSegmentation'),
    'HE_FGADR': Lesion(dir_name='Hemohedge_Masks', project_name='FGADRHaemorrhageSegmentation'),
    'SE_FGADR': Lesion(dir_name='SoftExudate_Masks', project_name='FGADRSoftExudatesSegmentation'),
    'MA_FGADR': Lesion(dir_name='Microaneurysms_Masks', project_name='FGADRMicroaneurysmsSegmentation'),
    'Vessel_DRIVE': Lesion(dir_name='', project_name='DRIVE_VesselSegmentation'),
    'Vessel_HRF': Lesion(dir_name='', project_name='HRF_VesselSegmentation'),
    'Vessel_CHASEDB1': Lesion(dir_name='', project_name='CHASEDB1_VesselSegmentation')
}

def multigen(gen_func):
    class _multigen(object):
        def __init__(self, *args, **kwargs):
            self.__args = args
            self.__kwargs = kwargs
        def __iter__(self):
            return gen_func(*self.__args, **self.__kwargs)
    return _multigen

def make_grid(shape, window=256, min_overlap=32):
    """
        Return Array of size (N,4), where N - number of tiles,
        2nd axis represente slices: x1,x2,y1,y2 
    """
    x, y = shape
    nx = x // (window - min_overlap) + 1
    x1 = np.linspace(0, x, num=nx, endpoint=False, dtype=np.int64)
    x1[-1] = x - window
    x2 = (x1 + window).clip(0, x)
    ny = y // (window - min_overlap) + 1
    y1 = np.linspace(0, y, num=ny, endpoint=False, dtype=np.int64)
    y1[-1] = y - window
    y2 = (y1 + window).clip(0, y)
    slices = np.zeros((nx,ny, 4), dtype=np.int64)
    
    for i in range(nx):
        for j in range(ny):
            slices[i,j] = x1[i], x2[i], y1[j], y2[j]    
    return slices.reshape(nx*ny,4)
    

def minmax_normalize(img, norm_range=(0, 1), orig_range=(0, 255)):
    # range(0, 1)
    norm_img = (img - orig_range[0]) / (orig_range[1] - orig_range[0])
    # range(min_value, max_value)
    norm_img = norm_img * (norm_range[1] - norm_range[0]) + norm_range[0]
    return norm_img
    

def get_datapath(img_path: Union[Path, Tuple[Path]], mask_path: Union[Path, Tuple[Path]], lesion_type: str = 'EX'):
    if lesion_type.split('_')[0] =='Vessel':
        full_img_paths = list(img_path.glob('*.jpg'))
        full_mask_paths = list(mask_path.glob('*.jpg'))
        return sorted(full_img_paths), sorted(full_mask_paths)

    if len(lesion_type.split('_')) == 1:
        lesion_path = lesion_dict[lesion_type].dir_name
        img_posfix = '.jpg'
        mask_posfix = '_' + lesion_type + '.tif'
        mask_names = os.listdir(os.path.join(mask_path, lesion_path))

        mask_ids = list(map(lambda mask: re.sub(
            mask_posfix, '', mask), mask_names))

        restored_name = list(map(lambda x: x + img_posfix, mask_ids))
        full_img_paths = list(
            map(lambda x: Path(os.path.join(img_path, x)), restored_name))
        full_mask_paths = list(
            map(lambda x: Path(os.path.join(mask_path, lesion_path, x)), mask_names))
        return sorted(full_img_paths), sorted(full_mask_paths)

    if lesion_type.split('_')[1] == 'FGADR':
        lesion_path = lesion_dict[lesion_type].dir_name
        full_img_paths = list(img_path.glob('*.png'))
        full_mask_paths = list((mask_path / lesion_path).glob('*.png'))
        return sorted(full_img_paths), sorted(full_mask_paths)

    if lesion_type.split('_')[1] == 'DDR':
        lesion_path = lesion_dict[lesion_type].dir_name
        if isinstance(img_path, tuple):
            train_img = list(img_path[0].glob('*.jpg'))
            train_mask = list((mask_path[0] / lesion_path).glob('*.tif'))

            valid_img = list(img_path[1].glob('*.jpg'))
            valid_mask = list((mask_path[1] / lesion_path).glob('*.tif'))
            return (sorted(train_img), sorted(valid_img)), (sorted(train_mask), sorted(valid_mask))
        else:
            img = list(img_path.glob('*.jpg'))
            mask = list((mask_path / lesion_path).glob('*.tif'))
            return sorted(img), sorted(mask)

def save_output(pred_masks: np.ndarray, out_path: Path):
    # Rescale to 0-255 and convert to uint8
    rescaled = (255.0 / (pred_masks.max() + np.finfo(float).eps) *
                (pred_masks - pred_masks.min())).astype(np.uint8)

    im = Image.fromarray(rescaled)
    im.save(out_path)
    print(f'[INFO] saved {out_path.name} to disk')


def log_pretty_table(col_names, row_data):
    x = PrettyTable()

    x.field_names = col_names
    for row in row_data:
        x.add_row(row)

    print(x)

def distributed_cmd_run(
    worker_fn: Callable, distributed: bool = True, *args, **kwargs
) -> None:
    """
    Distributed run
    Args:
        worker_fn: worker fn to run in distributed mode
        distributed: distributed flag
        args: additional parameters for worker_fn
        kwargs: additional key-value parameters for worker_fn
    """
    distributed_params = get_distributed_params()
    local_rank = distributed_params["local_rank"]
    world_size = distributed_params["world_size"]

    if distributed and torch.distributed.is_initialized():
        warnings.warn(
            "Looks like you are trying to call distributed setup twice, "
            "switching to normal run for correct distributed training."
        )

    if (
        not distributed
        or torch.distributed.is_initialized()
        or world_size <= 1
    ):
        worker_fn(*args, **kwargs)
    elif local_rank is not None:
        torch.cuda.set_device(int(local_rank))

        torch.distributed.init_process_group(
            backend="gloo", init_method="env://"
        )
        worker_fn(*args, **kwargs)
    else:
        workers = []
        try:
            for local_rank in range(torch.cuda.device_count()):
                rank = distributed_params["start_rank"] + local_rank
                env = get_distributed_env(local_rank, rank, world_size)
                cmd = [sys.executable] + sys.argv.copy()
                workers.append(subprocess.Popen(cmd, env=env))
            for worker in workers:
                worker.wait()
        finally:
            for worker in workers:
                worker.kill()
