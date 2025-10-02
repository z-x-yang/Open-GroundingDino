# https://warwick.ac.uk/fac/cross_fac/tia/data/nuclick/
# https://arxiv.org/pdf/2005.14511

import os, glob, sys
import numpy as np
import pandas as pd
from PIL import Image
sys.path.append('./')
from utils.cfg import VLDATA_RAW

# lymphocyte instance mask
if __name__ == '__main__':
    image_dir = f'{VLDATA_RAW}/ihc_nuclick/IHC/images'
    label_dir = f'{VLDATA_RAW}/ihc_nuclick/IHC/masks'
    for sub in ['Train', 'Validation']:
        _image_dir = f'{image_dir}/{sub}'
        _label_dir = f'{label_dir}/{sub}'
        for labelpath in glob.glob(f'{_label_dir}/*.npy'):
            imagepath = os.path.join(_image_dir, os.path.basename(labelpath)).replace('.npy', '.png')
            img = Image.open(imagepath).convert('RGB')
            width, height = img.size
            mask = np.load(labelpath)
            print(np.unique(mask))
            assert mask.shape == (height, width), (mask.shape, height, width)