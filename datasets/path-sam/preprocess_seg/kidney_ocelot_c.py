# https://ocelot2023.grand-challenge.org/datasets/
import sys
sys.path.append('../')
sys.path.append('./')
import os
import glob
import numpy as np
from tqdm import tqdm
from PIL import Image
import pandas as pd
from utils.cfg import VLDATA_RAW
from preprocess.tile_utils import crop_img_mask


def crop_with_cell_centroid(outdir, imgf, tumor_cell_pts, tile_hw=336):
    os.makedirs(f'{outdir}/1', exist_ok=True)
    os.makedirs(f'{outdir}/2', exist_ok=True)
    
    def cell_in_bbox(x0,x1,y0,y1):
        for pt in tumor_cell_pts:
            x,y = pt
            if x>=x0 and x<x1 and y>=y0 and y<y1:
                return True
        return False
    
    imgid = imgf.split('/')[-1].split('.')[0]
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
            if cell_in_bbox(x0,x1,y0,y1):
                Image.fromarray(img).save(f'{outdir}/2/{imgid}_{x0}_{y0}.png')
            else:
                Image.fromarray(img).save(f'{outdir}/1/{imgid}_{x0}_{y0}.png')

def crop_patch():
    image_dir = f'{VLDATA_RAW}/ocelot_processed/ocelot2023_v1.0.1/images/test'
    anno_dir = f'{VLDATA_RAW}/ocelot_processed/ocelot2023_v1.0.1/annotations/test'
    outdir = f'{VLDATA_RAW}/ocelot_processed/test_patches'

    # Tissue: Background (BG, index 1), Cancer Area (CA, index 2), and Unknown (not labeled, index 255)
    for annof in tqdm(glob.glob(f'{anno_dir}/tissue/*.png')):
        imgname = annof.split("/")[-1].split('.')[0]
        imgf = f'{image_dir}/tissue/{imgname}.jpg'
        mask = np.array(Image.open(annof))
        mask[mask == 255] = 1
        crop_img_mask(whole_img=np.array(Image.open(imgf)),
                      whole_mask=mask,
                      out_dir=f'{outdir}/tissue',
                      wsi_mark=imgname,
                      label_def_dict={1: 'Background', 2: 'Cancer Area'},
                      label_thresh_dict={1: 0.95, 2: 0.7},
                      tile_hw=336,
                      save_mask=False)
    # Cell: Background Cell (BC, index 1) and Tumor Cell (TC, index 2)
    for annof in tqdm(glob.glob(f'{anno_dir}/cell/*.csv')):
        imgf = f'{image_dir}/cell/{annof.split("/")[-1].replace("csv", "jpg")}'
        if os.path.exists(imgf):            
            try:
                df = pd.read_csv(annof, header=None)
                cell_pts = df.values
                tumor_cell_pts = cell_pts[cell_pts[:, 2] == 2][:, :2]
                crop_with_cell_centroid(outdir=f'{outdir}/cell',
                                        imgf=imgf,
                                        tumor_cell_pts=tumor_cell_pts)
            except:
                print(imgf, annof, "Failed")

