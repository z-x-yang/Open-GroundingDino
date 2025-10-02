# https://github.com/lifangda01/AdaptiveSupervisedPatchNCE?tab=readme-ov-file



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
    _dir = f'{VLDATA_RAW}/mist'
    pair_cnt = 0
    for subfolder in 'PR ER HER2 Ki67'.split():
        for split in 'train val'.split():
            folderA = f'{_dir}/{subfolder}/TrainValAB/{split}A'
            folderB = f'{_dir}/{subfolder}/TrainValAB/{split}B'
            for imgpathA in glob.glob(f'{folderA}/*.jpg'):
                imgpathB = imgpathA.replace(f'{folderA}', f'{folderB}')
                imgA = np.array(Image.open(imgpathA))
                imgB = np.array(Image.open(imgpathB))
                assert imgA.shape == imgB.shape, f"Shape mismatch: {imgA.shape} vs {imgB.shape}"
                pair_cnt += 1
    print(f"Total image pairs: {pair_cnt}") #Total image pairs: 21295