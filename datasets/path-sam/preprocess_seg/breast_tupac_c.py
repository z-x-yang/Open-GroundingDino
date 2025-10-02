# https://tupac.grand-challenge.org/

import os, glob
from utils.cfg import VLDATA_RAW
from PIL import Image
import pandas as pd
import numpy as np
from tqdm import tqdm 




def crop_with_mitoses_centroid(outdir, imgf, mitoses_pts, tile_hw=336):
    def mitoses_in_bbox(x0,x1,y0,y1):
        for mitoses_pt in mitoses_pts:
            y, x = mitoses_pt
            if x>=x0 and x<x1 and y>=y0 and y<y1:
                return True
        return False
    
    imgid = '-'.join(imgf.split('.')[0].split('/')[-2:])
    image = np.array(Image.open(imgf))
    height, width = image.shape[:2]
    num_x = width//tile_hw
    num_y = height//tile_hw
    print(f'{num_x*num_y} patches in total')
    for y in range(num_y):
        for x in range(num_x):
            y0, y1 = y*tile_hw, (y+1)*tile_hw
            x0, x1 = x*tile_hw, (x+1)*tile_hw
            img = image[y0:y1, x0:x1]
            if mitoses_in_bbox(x0,x1,y0,y1):
                Image.fromarray(img).save(f'{outdir}/1/{imgid}_{x0}_{y0}.png')
            else:
                Image.fromarray(img).save(f'{outdir}/0/{imgid}_{x0}_{y0}.png')

def rawimg_to_patch():
    base_dir = f'{VLDATA_RAW}/TUPAC-mitoses'
    imgfs = glob.glob(f'{base_dir}/mitoses_image/*/*.tif')
    patch_dir = f'{base_dir}/process_patch'
    os.makedirs(f'{patch_dir}/0', exist_ok=True)
    os.makedirs(f'{patch_dir}/1', exist_ok=True)

    for imgf in tqdm(imgfs):
        mitoses_ground_truth = imgf.replace('mitoses_image', 'mitoses_ground_truth')
        mitoses_ground_truth = mitoses_ground_truth.replace('.tif', '.csv')
        if os.path.exists(mitoses_ground_truth):
            df = pd.read_csv(mitoses_ground_truth, header=None)
            mitoses_pts = df.values #(N, 2), row and column
        crop_with_mitoses_centroid(patch_dir, imgf, mitoses_pts)

        