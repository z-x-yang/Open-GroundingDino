# https://www.nature.com/articles/s41597-022-01450-y
import pandas as pd
import glob
import numpy as np
from PIL import Image
import math
import openslide
from tqdm import tqdm

import os, sys, json
sys.path.append('./')
from utils.cfg import VLDATA_RAW, VLDATA_PROCESS
from preprocess.tile_utils import crop_img_mask_to_caption


_RAW_DIR = f'{VLDATA_RAW}/HunCRC'
_PROC_DIR = f'{VLDATA_RAW}/HunCRC_processed'
#The 200 digital slides, after pre-processing, resulted in 101,389 patches.
# A single patch is a 512 × 512 pixel image, covering 248 × 248 μm2 tissue area.



def load_huncrc_patch_label():
    keys = [
        'normal',

        'inflammation',
        'resection_edge',
        'tumor_necrosis',
        'artifact', #should be removed

        'adenocarcinoma',
        'suspicious_for_invasion',
        'lymphovascular_invasion',
        
        'lowgrade_dysplasia',
        'highgrade_dysplasia',]
    base_dir = f'{_RAW_DIR}/patches'
    df = pd.read_csv(f'{base_dir}/200_labels.csv')
    df['imgname'] = df['fname'].apply(lambda x: x.split('/')[-1])
    df = df[['imgname']+keys]

    imgfs = glob.glob(f'{base_dir}/200/*.jpg') #1246 patches from 200 slides
    gt_list = []
    for imgf in imgfs:
        imgname = imgf.split('/')[-1]
        row = df.loc[df['imgname'] == imgname]
        labels = row.values[0].tolist()[1:]
        assert np.sum(labels)==1
        gt = labels.index(1)
        gt_list.append(gt)
    print(set(gt_list)) #(0,8,9)
    breakpoint()

def _decode_maskf(maskf):
    # 200_highgrade_dysplasia_(8.00,67549,318494,45179,32925)-mask.png
    maskname = maskf.split('/')[-1]
    wsid = maskname.split('_')[0]
    category = '_'.join(maskname.split('_(')[0].split('_')[1:])
    coords = maskname.split('_(')[1].split(')')[0].split(',')
    downscale, x, y, w, h = int(float(coords[0])), int(coords[1]), int(coords[2]), int(coords[3]), int(coords[4])
    return wsid, category, downscale, x, y, w, h



mask_classes = ['inflammation', 'highgrade_dysplasia', 'suspicious_for_invasion', 'dysplasia',
    'adenocarcinoma', 'tumor_necrosis']


def save_huncrc_wsiregion_from_mask(wsi_dir, image_region_dir, mask_region_dir):
    os.makedirs(image_region_dir, exist_ok=True)

    maskfs = glob.glob(f'{mask_region_dir}/*.png')
    all_class = set()
    for maskf in maskfs:
        wsid, category, downscale, x, y, w, h = _decode_maskf(maskf)
        level = int(math.log2(downscale))
        all_class.add(category)
        if category in mask_classes:
            mask = Image.open(maskf)
            mask = np.array(mask) # 0 or 255
            # print(np.unique(mask))
            height, width = mask.shape[:2]
            assert (math.floor(h/downscale+0.5)==height and math.floor(w/downscale+0.5)==width)
            # print(h/downscale, height, w/downscale, width)
            # read image
            wsipath = f'{wsi_dir}/{wsid}.mrxs'
            imagef = os.path.basename(maskf).replace('-mask.png', '.png')
            try:
                wsi = openslide.OpenSlide(wsipath)
                image = wsi.read_region((x, y), level, (width, height)) #starting index at level 0; width, height at target level
                image = image.convert("RGB")
                image.save(f'{image_region_dir}/{imagef}')
                print(np.array(image).shape, mask.shape)
            except:
                print(wsipath)
    print(all_class)
    #'inflammation', 'highgrade_dysplasia', 'annotated', 'suspicious_for_invasion', 'dysplasia',
    # 'adenocarcinoma', 'tumor_necrosis', 'artifact', 'resection_edge'

def reset_wsi_readable(target_dir):
    source_dir = f'{_RAW_DIR}/slides'
    for slidid in range(1,201):
        slidid = "{:03}".format(slidid)
        os.makedirs(f'{target_dir}/{slidid}', exist_ok=True)

        sourcef = f'{source_dir}/{slidid}/{slidid}.mrxs'
        # os.system(f'ln -s {sourcef} {target_dir}/{slidid}.mrxs')
        # os.system(f'ln -s {source_dir}/{slidid} {target_dir}/')
        
        os.system(f'cp -r {source_dir}/{slidid} {target_dir}/')
        os.system(f'mv {target_dir}/{slidid}/{slidid}.mrxs {target_dir}/')
        print(slidid)

def crop_save(image_region_dir, mask_region_dir, patch_dir, jsonf):
    data_list = []
    maskfs = glob.glob(f'{mask_region_dir}/*.png')
    for maskf in tqdm(maskfs):
        wsid, category, downscale, x, y, w, h = _decode_maskf(maskf)
    
        if category in mask_classes:
            mask = Image.open(maskf)
            mask = np.array(mask) # 0 or 255
            assert set(np.unique(mask)).issubset(set([0, 255]))
            imagef = os.path.basename(maskf).replace('-mask.png', '.png')
            imgf = f'{image_region_dir}/{imagef}'
            img = np.array(Image.open(imgf))
            assert img.shape[:2] == mask.shape[:2]
            # Optimize crop size
            standard_hw = 336
            height, width = img.shape[:2]
            num_width, num_height = width//standard_hw+1, height//standard_hw+1
            # num_width, num_height = max(num_width, 1), max(1, num_height)
            if max(num_width, num_height)<=2:
                tile_hw = min(height//num_height, width//num_width)
            else:
                tile_hw = standard_hw
            num_tile = (height//tile_hw)*(width//tile_hw)
            print(height, width, tile_hw, '#',num_tile)
            # Crop with foreground mask
            results = crop_img_mask_to_caption(whole_img=img,
                        whole_mask=mask[:,:,None],
                        out_dir=patch_dir,
                        wsi_mark=imagef[:-4],
                        label_def_dict={255: category}, 
                        label_thresh_dict={255: 0.5},
                        tile_hw=tile_hw)
            results = [r for r in results if r]
            print(len(results))
            data_list+= results
    print(len(data_list))
    with open(jsonf, 'w') as json_file:
        json.dump(data_list, json_file, indent=4)   


if __name__ == '__main__':
    # load_huncrc_patch_label()
    # 1. Rearange WSI file to make it readable
    wsi_dir =  f'{_PROC_DIR}/slides'
    if not os.path.exists(wsi_dir):
        reset_wsi_readable()
    # 2. Save image region according to region mask
    image_region_dir = f'{_PROC_DIR}/regions'
    if not os.path.exists(image_region_dir):
        save_huncrc_wsiregion_from_mask(wsi_dir = wsi_dir,
                                    image_region_dir=image_region_dir,
                                    mask_region_dir=f'{_RAW_DIR}/masks/masks')
    # 3. Crop region image/mask to patch
    patch_dir = f'{VLDATA_PROCESS}/huncrc_cropped'

    os.makedirs(patch_dir, exist_ok=True)
    crop_save(
        image_region_dir=image_region_dir,
        mask_region_dir=f'{_RAW_DIR}/masks/masks',
        patch_dir=patch_dir,
        jsonf = f'{VLDATA_PROCESS}/colon_huncrc.json')
    # breakpoint()