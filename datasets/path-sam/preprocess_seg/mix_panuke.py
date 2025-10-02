# https://www.kaggle.com/datasets/andrewmvd/cancer-inst-segmentation-and-classification/data

import numpy as np
import os
from PIL import Image
import cv2
import json

import sys
sys.path.append('./')
from utils.cfg import VLDATA_RAW, VLDATA_PROCESS
from utils.draw import draw_image, COLORS

_RAW_DIR = f'{VLDATA_RAW}/panuke'
_SUB_DIRS = [f'{_RAW_DIR}/part{n}' for n in [1,2,3]]

_OUT_DIR = f'{VLDATA_RAW}/panuke_processed'
os.makedirs(_OUT_DIR, exist_ok=True)

_VIS_DIR = f'{VLDATA_PROCESS}/visualization/panuke'
os.makedirs(_VIS_DIR, exist_ok=True)

def save_img_mask_png(images, masks, output_dir, save_mask=False):
    os.makedirs(output_dir, exist_ok=True)
    num_image = len(images)
    for n, (image, mask) in enumerate(zip(images, masks)):
        # cv2.imwrite(f'{output_dir}/{n}_BGR.jpeg', image)
        pil_image = Image.fromarray(image.astype('uint8'))
        pil_image.save(f'{output_dir}/{n}_RGB.jpeg')
        if save_mask:
            for n_ch in range(mask.shape[-1]):
                cur_mask = mask[:,:,n_ch]
                print(np.unique(cur_mask))
                cur_mask = (255*cur_mask//cur_mask.max()).astype('uint8')
                cv2.imwrite(f'{output_dir}/{n}_mask{n_ch}.jpeg', cur_mask)
        print(f'{n}/{num_image}')


_CH_LABEL = {
    0: 'Neoplastic cells', 
    1: 'Inflammatory', 
    2: 'Connective/Soft tissue cells', 
    3: 'Dead Cells', 
    4: 'Epithelial', 
    # 5: 'Background'
}

def visualize_cell(imgf, mask):
    mask = np.uint8(mask)
    image = np.array(Image.open(imgf)).astype('uint8')
    for n_ch, text in _CH_LABEL.items():
        cur_mask = mask[:,:,n_ch]
        for label in np.unique(cur_mask):
            if label>0:
                binary_mask = (cur_mask==label).astype('uint8')
                contours, hierarchy = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                image = draw_image(image, contours, [text]*len(contours), color=COLORS[n_ch])
   
    cv2.imwrite(f'{_VIS_DIR}/{os.path.basename(imgf)}',  cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

def save_panuke_json(jsonf):
    datalist = []
    
    for part, indir in enumerate(_SUB_DIRS):
        image_file = f'{indir}/Images/images.npy'
        type_file = f'{indir}/Images/types.npy'
        mask_file = f'{indir}/Masks/masks.npy'

        images = np.load(image_file) #(N, 256, 256, 3)
        organs = np.load(type_file) #(N,) texts
        print(np.unique(organs))
        masks = np.load(mask_file) #(N, 256, 256, 6) [0,1,2,......] instance label
        # masks = (masks>0).astype('uint8')
        # print(np.unique((masks>0).sum(axis=-1)))
        # print((masks.sum(axis=-1)==1).sum()/(masks.shape[0]*masks.shape[1]*masks.shape[2]))

        # Combine the mask
        num_image = len(images)
        num_label = len(_CH_LABEL)
        height, width = images.shape[1:3]
        num_pixel = height*width
        for n, (organ, mask) in enumerate(zip(organs, masks)):
            imgpath = f'{_OUT_DIR}/p{part}_{n}.png'
            # visualize_cell(imgpath, mask)
            # continue
            assert os.path.exists(imgpath), imgpath
            labels = {}
            counts = {}
            for n_ch in range(num_label):
                label = _CH_LABEL[n_ch]
                cur_mask = mask[:,:,n_ch]
                instance_idx = np.unique(cur_mask)
                count = sum(instance_idx>0)
                
                percent = (cur_mask>0).sum()/num_pixel
                labels.update({label:round(percent,4)})
                counts.update({label:float(count)})
            datalist.append({
                'image': imgpath,
                'organ': organ,
                'label': labels,
                'count': counts
            })

    print(len(datalist))
    with open(jsonf, 'w') as json_file:
        json.dump(datalist, json_file, indent=4)

def save_image_png():
    for part, indir in enumerate(_SUB_DIRS):
        image_file = f'{indir}/Images/images.npy'
        images = np.load(image_file) #(N, 256, 256, 3)
        for n, image in enumerate(images):
            pil_image = Image.fromarray(image.astype('uint8'))
            pil_image.save(f'{_OUT_DIR}/p{part}_{n}.png')
            print(f'{n}/{len(images)}')



if __name__ == '__main__':
    # save_image_png()
    save_panuke_json(jsonf=f'{VLDATA_PROCESS}/mix_panuke.json')
