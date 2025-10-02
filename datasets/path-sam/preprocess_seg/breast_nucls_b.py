# https://sites.google.com/view/nucls
import numpy as np
import os, json
import pandas as pd
from PIL import Image

import sys
sys.path.append('./')
from utils.cfg import VLDATA_RAW, VLDATA_PROCESS

_RAW_DIR = f'{VLDATA_RAW}/nucls'
_LABEL_DEF = {
    253: 'fov',#remove
    1: 'tumor', 
    2: 'fibroblast',
    3: 'lymphocyte',
    4: 'plasma cell',
    5: 'macrophage',
    6: 'mitotic figure',
    7: 'vascular endothelium',
    8: 'myoepithelium',
    9: 'apoptotic body',
    10: 'neutrophil',
    11: 'ductal epithelium',
    12: 'eosinophil',
    99: 'unlabled',#remove
    0: 'unknown'#remove
}
"""
.csv file
> raw_classification: raw class (13 total)
> main_classification: nucleus class (7 total)
> super_classification: nucleus superclass (4 total)
> type: rectangles v.s. polylines
> xmin / ymin / xmax / ymax: extent of the nucleus
> coords_x / coords_y: comma-separated boundary
mask file
Mask images are provided for convenient use with machine learning analyses. 
The first channel in each mask image encodes the class labels found here.
The product of the second and third channels encode the unique instance label for each nucleus.
The fov area (gray) is included in the class table and first channel of the mask.
This file contains the nucleus label encoding, including a special 'fov' code encoding the intended annotation region.
"""

def read_images_masks():
    imagefs, maskfs, labels = [], [], []
    for subfolder in ['train', 'eval']:
        imgdir = f'{_RAW_DIR}/{subfolder}/rgb'
        csvdir = f'{_RAW_DIR}/{subfolder}/csv'
        maskdir = f'{_RAW_DIR}/{subfolder}/mask'
        for maskf in os.listdir(maskdir):
            try:
                casemask = maskf[:-4]
                caseimg = casemask.split('#_')[-1]
                if len(caseimg.split('_')[0].split('-'))==6:
                    tcgacase = caseimg.split('_')[0]
                    tcgacase = '-'.join(tcgacase.split('-')[:3]+[tcgacase.split('-')[-1]])
                    caseimg = '_'.join([tcgacase]+caseimg.split('_')[1:]) 
                imgf = f'{imgdir}/{caseimg}.png'
                maskf = f'{maskdir}/{casemask}.png'
                csvf = f'{csvdir}/{casemask}.csv'
                img = np.array(Image.open(imgf))
                mask = np.array(Image.open(maskf))[:,:,0]
                df = pd.read_csv(csvf)
                print(img.shape, mask.shape) #, np.unique(mask).shape, len(df))
                labels += np.unique(mask).tolist()
                imagefs.append(imgf)
                maskfs.append(maskf)
            except:
                pass

    print(len(imagefs)) #1744, 
    print(set(labels))
    return imagefs, maskfs

if __name__ == '__main__':
    imagefs, maskfs = read_images_masks()

    # Mask to caption
    jsonf = f'{VLDATA_PROCESS}/breast_nucls.json'
    datalist = []
    for imgf, maskf in zip(imagefs, maskfs):
        mask = np.array(Image.open(maskf))[:,:,0]
        num_pixel = mask.shape[0]*mask.shape[1]
        caption = {}
        for label, deff in _LABEL_DEF.items():
            if 1<=label<=12:
                percent = np.sum(mask==label)/num_pixel
                caption[deff] = round(percent, 4)
        datalist.append({'image': imgf, 'label': caption})
    print(len(datalist))
    with open(jsonf, 'w') as json_file:
        json.dump(datalist, json_file, indent=4)