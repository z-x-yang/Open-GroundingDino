# https://www.kaggle.com/datasets/sbilab/segpc2021dataset


# segment each instance of the cell (nucleus + cytoplasm) of interest. Please assign the following labels to each class:
# Background: '0'
# Cytoplasm: '1'
# Nucleus: '2'
# 1+2 = Cell

import os, glob, sys
import numpy as np
from PIL import Image
sys.path.append('./')
from utils.cfg import VLDATA_RAW

if __name__ == '__main__':
    datadir = f'{VLDATA_RAW}/segpc'
    Nucleus_VALUE = 40
    Cytoplasm_VALUE = 20
    for sub in ['train/train/train', 'validation/validation', 'test']:
        _dir = f'{datadir}/{sub}'
        imgpaths = glob.glob(f'{_dir}/x/*.bmp')
        print(sub, len(imgpaths))
        for imgpath in imgpaths:
            name = os.path.basename(imgpath).replace('.bmp', '')
            labelpaths = glob.glob(f'{_dir}/y/{name}_*.bmp')
            for labelpath in labelpaths: #instances
                img = np.array(Image.open(imgpath))
                label = np.array(Image.open(labelpath))
                assert img.shape[:2] == label.shape[:2], (img.shape, label.shape)
                assert np.all(np.unique(label) == [0, Cytoplasm_VALUE, Nucleus_VALUE]), np.unique(label)
                # cell_mask = np.zeros_like(label, dtype=np.uint8)
                # cell_mask[label == Cytoplasm_VALUE] = 1
                # cell_mask[label == Nucleus_VALUE] = 1
                # nuclei_mask = np.zeros_like(label, dtype=np.uint8)
                # nuclei_mask[label == Nucleus_VALUE] = 1