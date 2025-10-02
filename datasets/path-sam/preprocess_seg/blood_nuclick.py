# https://warwick.ac.uk/fac/cross_fac/tia/data/nuclick/
# https://arxiv.org/pdf/2005.14511
# https://github.com/navidstuv/NuClick


import os, glob, sys
import numpy as np
import pandas as pd
from PIL import Image
sys.path.append('./')
from utils.cfg import VLDATA_RAW

# white blood cell instance mask
if __name__ == '__main__':
    image_dir = [f'{VLDATA_RAW}/Hemato_Data/Train/images',f'{VLDATA_RAW}/Hemato_Data/Validation/images']
    label_dir = [f'{VLDATA_RAW}/Hemato_Data/Train/masks',f'{VLDATA_RAW}/Hemato_Data/Validation/masks']
    for _image_dir, _label_dir in zip(image_dir, label_dir):
        for labelpath in glob.glob(f'{_label_dir}/*.png'):
            imagepath = os.path.join(_image_dir, os.path.basename(labelpath)).replace('_mask.png', '.png')
            img = Image.open(imagepath).convert('RGB')
            mask = np.array(Image.open(labelpath))
            width, height = img.size
            print(np.unique(mask))
            assert mask.shape == (height, width), (mask.shape, height, width)