
# https://github.com/ieee8023/countception/tree/master?tab=readme-ov-file
# https://github.com/ieee8023/countception/tree/master/adipocyte_data
import os, glob, sys
import numpy as np
import pandas as pd
import cv2
from PIL import Image
sys.path.append('./')
from utils.cfg import VLDATA_RAW
from utils.draw import draw_contour_from_mask
# adipoctye centroid
if __name__ == '__main__':
    image_dir = f'{VLDATA_RAW}/adipocyte_data/images'
    label_dir = f'{VLDATA_RAW}/adipocyte_data/annotations'
    vis_dir = f'{VLDATA_RAW}/adipocyte_data/vis'
    os.makedirs(vis_dir, exist_ok=True)
    for labelpath in glob.glob(f'{label_dir}/*.jpeg'):
        imgname = os.path.basename(labelpath)
        imagepath = os.path.join(image_dir, imgname)
        img = Image.open(imagepath).convert('RGB')
        width, height = img.size
        mask = np.array(Image.open(labelpath).convert('L'))
        print(np.unique(mask))
        assert mask.shape == (height, width), (mask.shape, height, width)
        centroid_mask = (mask>75).astype('uint8')
        # img = draw_contour_from_mask(np.array(img), mask, text='', color=(0,255,0))
        # cv2.imwrite(f'{vis_dir}/{imgname}', img)

