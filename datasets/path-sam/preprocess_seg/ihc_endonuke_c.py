# https://endonuke.ispras.ru/
# https://github.com/ispras/endometrium-dataset-analysis/blob/master/endoanalysis/datasets.py


import os, glob, sys
import numpy as np
import pandas as pd
from PIL import Image
sys.path.append('./')
from utils.cfg import VLDATA_RAW
import yaml
import cv2
from matplotlib import patches, cm


"""
Class labels:
0 for the stroma
1 for the epithelium
2 for the the other nuclei
"""
if __name__ == '__main__':
    image_dir = f'{VLDATA_RAW}/endonuke/data/dataset/images'
    label_dir = f'{VLDATA_RAW}/endonuke/data/dataset/labels/bulk'
    label_cohorts = [f'ptg{x}' for x in [1,2,3]] +[f'stud{x}' for x in [1,2,3,4]]
    total_cnt = {0:0, 1:0, 2:0}
    for cohort in label_cohorts:
        _label_dir = f'{label_dir}/{cohort}'
        for labelpath in glob.glob(f'{_label_dir}/*.txt'):
            imagepath = os.path.join(image_dir, os.path.basename(labelpath).replace('.txt', '.png'))
            image = Image.open(imagepath).convert('RGB')
            width, height = image.size
            with open(labelpath, "r") as f:
                for line in f:
                    line = line.strip('\n')
                    centroid_x, centroid_y, class_label = line.split()
                    centroid_x, centroid_y, class_label = int(centroid_x), int(centroid_y), int(class_label)
                    assert 0<=centroid_x<width and 0<=centroid_y<height, (centroid_x, centroid_y, width, height)
                    assert class_label in [0, 1, 2], class_label
                    total_cnt[class_label] += 1
            print(total_cnt) #{0: 170996, 1: 37722, 2: 1701}