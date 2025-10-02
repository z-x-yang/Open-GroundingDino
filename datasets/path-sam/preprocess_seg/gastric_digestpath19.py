# https://digestpath2019.grand-challenge.org/
import sys
sys.path.append('../')
sys.path.append('./')

import xmltodict
import glob, os
import numpy as np
from tqdm import tqdm
import cv2


from utils.cfg import VLDATA_RAW
from PIL import Image
Image.MAX_IMAGE_PIXELS = None 
from preprocess.tile_utils import crop_img_mask_wcountour

# unzip DigestPath19/Colonoscopy_tissue_segment_dataset-20240817T203817Z-001.zip -d digestpath19_processed/
# unzip DigestPath19/Colonoscopy_tissue_segment_dataset-20240817T203817Z-002.zip -d digestpath19_processed/
# unzip DigestPath19/Signet_ring_cell_dataset-20240817T203348Z-001.zip -d digestpath19_processed/

base_dir = f'{VLDATA_RAW}/digestpath19_processed'
tissue_patch_dir = f'{base_dir}/tissue_patch336'

def crop_patches(group='pos'):
    if group=='neg':
        imgfs = glob.glob(f'{base_dir}/Colonoscopy_tissue_segment_dataset/tissue-train-neg/*.jpg')
        for imgf in tqdm(imgfs):
            img = np.array(Image.open(imgf))
            imgname = imgf.split('/')[-1].split('.')[0]
            height, width = img.shape[:2]
            mask_bg = np.zeros((height, width, 1))
            crop_img_mask_wcountour(whole_img=img,
                            whole_mask=mask_bg,
                            out_dir=f'{tissue_patch_dir}_neg', 
                            wsi_mark=imgname,
                            label_def_dict={0: 'Benign', 1: 'malignent'},
                            label_thresh_dict={0: 0.95, 1: 0.7},
                            tile_hw= 336,
                            save_mask=False)
    elif group=='pos':
        maskfs = glob.glob(f'{base_dir}/Colonoscopy_tissue_segment_dataset/tissue-train-pos-v1/*_mask.jpg')
        for maskf in tqdm(maskfs):
            imgf = maskf.replace('_mask.jpg', '.jpg')
            imgname = imgf.split('/')[-1].split('.')[0]
            whole_img = np.array(Image.open(imgf))
            whole_mask = np.array(Image.open(maskf))[:,:,None]
            whole_mask = (whole_mask>128).astype(int)
            # print(np.unique(whole_mask), np.sum(whole_mask==1)/(336*336))
            crop_img_mask_wcountour(whole_img=whole_img,
                            whole_mask=whole_mask,
                            out_dir=f'{tissue_patch_dir}_pos', 
                            wsi_mark=imgname,
                            label_def_dict={0: 'Benign', 1: 'malignent'}, #, 
                            label_thresh_dict={0: 0.95, 1: 0.7}, #,
                            tile_hw= 336,
                            save_mask=False)

def load_sig_mask(image, maskf, height, width, patch_size, visf=''):
    with open(maskf) as xml_file:
        data = xmltodict.parse(xml_file.read())
    imgh, imgw = int(data['annotation']['size']['height']), int(data['annotation']['size']['width'])
    assert imgh==height and imgw==width, f'{maskf} {imgh} {imgw} {height} {width}'
    objects = data['annotation']['object']
    mask = np.zeros((imgh, imgw), dtype=np.uint8)
    visimg = image.copy()
    for obj in objects:
        if obj['name'] == 'ring_cell_cancer':
            x0 = int(obj['bndbox']['xmin'])
            y0 = int(obj['bndbox']['ymin'])
            x1 = int(obj['bndbox']['xmax'])
            y1 = int(obj['bndbox']['ymax'])
            mask[y0:y1, x0:x1] = 1
            cv2.rectangle(visimg, (x0, y0), (x1, y1), color=(0, 255, 0), thickness=2)
            print(f'{y1-y0} x {x1-x0} = {(y1-y0)*(x1-x0)/patch_size/patch_size} of the patch')
        else:
            continue
    if visf:
        Image.fromarray(visimg).save(visf)
    return mask
def crop_sig_patch(group='neg', patchsize=336):# read ring cells  bbox
    sig_patch_dir = f'{base_dir}/sig_patch_{patchsize}'

    assert group in ['pos', 'neg']
    sig_base_dir = f'{base_dir}/Signet_ring_cell_dataset/sig-train-{group}'
    imgfs = glob.glob(f'{sig_base_dir}/*.jpeg')
    for imgf in tqdm(imgfs):
        img = np.array(Image.open(imgf))
        imgname = imgf.split('/')[-1].split('.')[0]
        height, width = img.shape[:2]
        
        maskf = imgf.replace('.jpeg', '.xml')
        if os.path.exists(maskf):
            mask = load_sig_mask(img, maskf,height, width,
                                 patch_size=patchsize,
                                 visf=f'{base_dir}/visualize/{imgname}.png' if group=='pos' else '')
        else:
            mask = np.zeros((height, width, 1))
        
        crop_img_mask_wcountour(whole_img=img,
                        whole_mask=mask,
                        out_dir=f'{sig_patch_dir}_{group}', 
                        wsi_mark=imgname,
                        label_def_dict={0: 'no ring cells', 1: 'ring cells'},
                        label_thresh_dict={0: 1, 1: 0.1},
                        tile_hw= patchsize,
                        save_mask=False)

if __name__ == '__main__':
    # crop_patches(group='pos')
    # crop_patches(group='neg') 
    # crop_sig_patch(group='pos')          
    crop_sig_patch(group='neg')