# https://www.kaggle.com/datasets/aadimator/conic-challenge-dataset

import numpy as np
import pandas as pd
from PIL import Image

import os, sys, json
import cv2
sys.path.append('./')
from utils.cfg import VLDATA_RAW, VLDATA_PROCESS
from utils.draw import draw_image, COLORS

_RAW_DIR = f'{VLDATA_RAW}/CoNIC/data'
_IMG_DIR = f'{VLDATA_PROCESS}/CoNIC'
os.makedirs(_IMG_DIR, exist_ok=True)

_VIS_DIR = f'{VLDATA_PROCESS}/visualization/conic'
os.makedirs(_VIS_DIR, exist_ok=True)

GT_DEF = {
    1: "neutrophil",
    2: "epithelial",
    3: "lymphocyte",
    4: "plasma",
    5: "eosinophil",
    6: "connective"}

def visualize_cells(images, masks, df):
    for n, (image, mask) in enumerate(zip(images, masks)):
        count = df.iloc[n].to_dict()
        image = np.array(image).astype('uint8')
        mask_count = {}
        for n_ch, (gt, deff) in enumerate(GT_DEF.items()):
            instance_mask = mask[:,:,0] * (mask[:,:,1]==gt)
            instance_labels = np.unique(instance_mask)
            mask_count[deff] = (instance_labels>0).sum()
            for label in instance_labels:
                if label>0:
                    binary_mask = (instance_mask==label).astype('uint8')
                    contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    image = draw_image(image, contours, [deff]*len(contours), color=COLORS[n_ch])
        cv2.imwrite(f'{_VIS_DIR}/{n}.png',  cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        # print(mask_count, count)
        # mask_count and count are not the same, but the cell propotionl similar

if __name__ == '__main__':
    imgf = f'{_RAW_DIR}/images.npy'
    labelf = f'{_RAW_DIR}/labels.npy'
    countf = f'{_RAW_DIR}/counts.csv'
    df = pd.read_csv(countf)
    images = np.load(imgf)
    masks = np.load(labelf)
    print(images.shape, masks.shape) 
    #(4981, 256, 256, 3), (4981, 256, 256, 2)
    
    visualize_cells(images, masks, df)
    breakpoint()  
    # Masks: the first channel is the instance segmentation map and the second channel is the classification map. 
    height_width = masks.shape[1]*masks.shape[2]
    
    jsonf = f'{VLDATA_PROCESS}/colon_conic.json'
    if not os.path.exists(jsonf):
        datalist = []
        for n, (image, mask) in enumerate(zip(images, masks)):
            count = df.iloc[n].to_dict()
            imgpath = f'{_IMG_DIR}/{n}.png'
            Image.fromarray(image.astype('uint8')).save(imgpath)
            
            label = {}
            for gt, deff in GT_DEF.items():
                percent = (mask[:,:,1]==gt).sum()/height_width
                label[deff] = round(percent,4)
            datalist.append(
                {'image': imgpath,
                'label': label,
                'count': count}
            )
            print(n)
        print(len(datalist))
        
        with open(jsonf, 'w') as json_file:
            json.dump(datalist, json_file, indent=4)