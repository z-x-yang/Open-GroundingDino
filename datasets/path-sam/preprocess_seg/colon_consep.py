# https://www.sciencedirect.com/science/article/pii/S1361841519301045?via%3Dihub
# https://www.kaggle.com/datasets/karthikperupogu/consep


import os, glob, sys
import numpy as np
from PIL import Image
sys.path.append('./')
from utils.cfg import VLDATA_RAW
from scipy.io import loadmat

"""
Class values: 
1 = other
2 = inflammatory
3 = healthy epithelial
4 = dysplastic/malignant epithelial
5 = fibroblast
6 = muscle
7 = endothelial
"""
if __name__ == '__main__':
    imgdirs = [f'{VLDATA_RAW}/CoNSeP_raw/Train/Images', f'{VLDATA_RAW}/CoNSeP_raw/Test/Images']
    labeldirs = [f'{VLDATA_RAW}/CoNSeP_raw/Train/Labels', f'{VLDATA_RAW}/CoNSeP_raw/Test/Labels']

    for imgdir, labeldir in zip(imgdirs, labeldirs):
        for labelpath in glob.glob(f'{labeldir}/*.mat'):
            imagepath = os.path.join(imgdir, os.path.basename(labelpath).replace('.mat', '.png'))
            image_size = np.array(Image.open(imagepath)).shape
            annot = loadmat(labelpath)
            inst_map = annot['inst_map']
            inst_type = annot['inst_type'].squeeze()
            print(inst_map.shape, np.unique(inst_map))
            print(inst_type.shape, np.unique(inst_type))
            # Size
            assert inst_map.shape == image_size[:2], (inst_map.shape,  image_size)
            # number of instances
            assert len(np.unique(inst_map))-1 == np.max(inst_map) == inst_type.shape, (len(np.unique(inst_map)), np.max(inst_map), inst_type.shape)
            # class 
            assert np.all([clsstype in np.arange(1,8) for clsstype in np.unique(inst_type)]), np.unique(inst_type)
