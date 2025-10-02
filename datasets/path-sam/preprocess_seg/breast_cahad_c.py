#https://figshare.com/articles/dataset/BreCaHAD_A_Dataset_for_Breast_Cancer_Histopathological_Annotation_and_Diagnosis/7379186

import os, glob
import json
from utils.cfg import VLDATA_RAW
from PIL import Image
import numpy as np
from collections import defaultdict
from tqdm import tqdm



def pts_in_bbox(pts, height, width, x0,x1,y0,y1):
    #the values are between [0, 1] (divided by width and height of an image).
    for pt in pts:
        x = pt['x']* width
        y = pt['y']* height
        if x>=x0 and x<x1 and y>=y0 and y<y1:
            return True
    return False

def crop_image(imgf, label_centroids, outdir, tile_hw=336):
    imageid = imgf.split('/')[-1].split('.')[0]
    image = np.array(Image.open(imgf)) #RGB
    height, width = image.shape[:2]
    num_x = width//tile_hw
    num_y = height//tile_hw
    print(f'{num_x*num_y} patches in total')
    for y in range(num_y):
        for x in range(num_x):
            y0, y1 = y*tile_hw, (y+1)*tile_hw
            x0, x1 = x*tile_hw, (x+1)*tile_hw
            if pts_in_bbox(label_centroids['tumor'], height, width, x0,x1,y0,y1):
                Image.fromarray(image[y0:y1, x0:x1]).save(f'{outdir}/tumor/{imageid}_{x0}_{y0}.png')
            elif pts_in_bbox(label_centroids['non_tumor'], height, width, x0,x1,y0,y1):
                Image.fromarray(image[y0:y1, x0:x1]).save(f'{outdir}/non_tumor/{imageid}_{x0}_{y0}.png')
    

def rawimg_to_patch():
    base_dir = f'{VLDATA_RAW}/BreCaHAD'
    outdir = f'{VLDATA_RAW}/BreCaHAD_processed'
    os.makedirs(f'{outdir}/tumor', exist_ok=True)
    os.makedirs(f'{outdir}/non_tumor', exist_ok=True)
    label_2_keys = {'tumor':['mitosis', 'tumor', 'lumen'], 
                    'non_tumor':['apoptosis', 'non_tumor', 'lumen', 'non_lumen']}# lumen equals to tuble
    imgfs = glob.glob(f'{base_dir}/images/*.tif')
    for imgf in tqdm(imgfs):
        label_centroids = defaultdict(list)
        # print(image.format, image.size, image.mode)
        jsonf = imgf.replace('/images/', '/groundTruth/')
        jsonf = jsonf.replace('.tif', '.json')
        with open(jsonf, 'r') as file:
            mask = json.load(file)
        for label, keys in label_2_keys.items():
            for key in keys:
                label_centroids[label] += mask[key]
        crop_image(imgf, label_centroids, outdir)