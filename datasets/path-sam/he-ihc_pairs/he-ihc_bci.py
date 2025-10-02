# https://bupt-ai-cz.github.io/BCI/


import os, glob, sys
from collections import defaultdict
import json
import numpy as np
import cv2
from scipy.io import loadmat
from PIL import Image
sys.path.append('./')
from utils.draw import draw_image
from utils.cfg import VLDATA_RAW


if __name__ == '__main__':
    _dir = f'{VLDATA_RAW}/BCI_dataset'
    pair_cnt = 0
    error_cnt = 0
    for split in 'train test'.split():
        folderA = f'{_dir}/HE/{split}'
        folderB = f'{_dir}/IHC/{split}'
        for imgpathA in glob.glob(f'{folderA}/*.png'):
            imgpathB = imgpathA.replace(f'{folderA}', f'{folderB}')
            try:
                imgA = np.array(Image.open(imgpathA))
                imgB = np.array(Image.open(imgpathB))
                assert imgA.shape == imgB.shape, f"Shape mismatch: {imgA.shape} vs {imgB.shape}"
                pair_cnt += 1
            except:
                # print(f"Error processing pair: {imgpathA} and {imgpathB}")
                error_cnt += 1
    print(f"Total image pairs: {pair_cnt}") # 4188
    print(f"Total errors encountered: {error_cnt}") #685
