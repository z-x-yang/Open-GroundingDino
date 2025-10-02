# https://github.com/vqdang/hover_net/issues/5
# Instance segmentation dataset: cell and background
import os, glob, json
import numpy as np
from PIL import Image
from scipy.io import loadmat
import sys
sys.path.append('../')
sys.path.append('./')

from utils.cfg import VLDATA_RAW

def load_kumar():
    base_dir = f'{VLDATA_RAW}/hover/kumar'
    imglist = glob.glob(f'{base_dir}/train/Images/*.tif') + glob.glob(f'{base_dir}/test_diff/Images/*.tif') + glob.glob(f'{base_dir}/test_same/Images/*.tif')
    labellist = glob.glob(f'{base_dir}/train/Labels/*.mat') + glob.glob(f'{base_dir}/test_diff/Labels/*.mat') + glob.glob(f'{base_dir}/test_same/Labels/*.mat')
    imglist.sort()
    labellist.sort()
    print(len(imglist), len(labellist))
    for imgf, labelf in zip(imglist, labellist):
        assert os.path.basename(imgf).split('.')[0] == os.path.basename(labelf).split('.')[0]
        image = np.array(Image.open(imgf))
        label = loadmat(labelf)
        print(label.keys())
        mask = label['inst_map']
        print(image.shape, mask.shape, np.unique(mask))

if __name__ == '__main__':
    load_kumar()