# https://dakomura.github.io/SegPath/

import numpy as np
import os,glob
from tqdm import tqdm
from PIL import Image
import json

import sys
sys.path.append('./')
from utils.cfg import VLDATA_RAW, VLDATA_PROCESS
from preprocess.tile_utils import crop_img_mask_to_caption



_RAW_DIR = f'{VLDATA_RAW}/segpath'
_SUBDIRS = ['aSMA_SmoothMuscle', 'CD235a_RBC', 'CD3CD20_Lymphocyte', 'CD45RB_Leukocyte', 'ERG_Endothelium', 'MIST1_PlasmaCell', 'MNDA_MyeloidCell', 'panCK_Epithelium']
_IMG_SIZE = (984, 984)


_FG_THRESHOLD = {
    'Epithelium': 0.5,
    'SmoothMuscle': 0.5,
    'RBC': 0.1,
    'Endothelium':0.05,
    'Leukocyte':0.1,
    'Lymphocyte': 0.1,
    'MyeloidCell': 0.1,
    'PlasmaCell': 0.1
    
}

def backup_save_segpath(_dir=f'{_RAW_DIR}_processed'):
    data = {}
    for subdir in _SUBDIRS:
        sub = subdir.split('_')[-1]
        imgfs1 = glob.glob(f'{_dir}/{sub}/0/*.png')
        imgfs2 = glob.glob(f'{_dir}/{sub}/1/*.png')
        print(sub, len(imgfs1), len(imgfs2))

    for subdir in _SUBDIRS:
        sub = subdir.split('_')[-1]
        pos_imgfs = glob.glob(f'{_dir}/{sub}/1/*.png')
        neg_imgfs = []
        for subdir2 in _SUBDIRS:
            if subdir2 != subdir:
                sub2 = subdir2.split('_')[-1]
                neg_imgfs += glob.glob(f'{_dir}/{sub2}/*/*.png')
        print(sub, len(pos_imgfs), len(neg_imgfs))
        data.update({sub: {0: neg_imgfs, 1: pos_imgfs}})

    with open(f'{VLDATA_PROCESS}/segpath.json', 'w') as json_file:
        json.dump(data, json_file, indent=4)
    # SmoothMuscle 118257 6455
    # RBC 101568 2068
    # Lymphocyte 47795 1297
    # Leukocyte 96079 3141
    # Endothelium 42516 72
    # PlasmaCell 52831 93
    # MyeloidCell 56475 65
    # Epithelium 82289 23747


def crop_save(patch_dir, jsonf):
    data_list = []
    for subdir in _SUBDIRS:
        imgfs = glob.glob(f'{_RAW_DIR}/{subdir}/*_HE.png')
        sub = subdir.split('_')[-1]
        for imgf in tqdm(imgfs):
            imgid = '_'.join(os.path.basename(imgf).split('_')[:-1])
            maskf = imgf.replace('_HE', '_mask')
            mask = Image.open(maskf)
            img = Image.open(imgf)
            assert img.size == mask.size == _IMG_SIZE
            assert set(np.unique(mask)).issubset(set([0, 1]))
            results = crop_img_mask_to_caption(whole_img=np.array(img),
                        whole_mask=np.array(mask)[:,:,None],
                        out_dir=f'{patch_dir}/{sub}',
                        wsi_mark=imgid,
                        label_def_dict={1: sub}, 
                        tile_hw=_IMG_SIZE[0]//2)
            data_list+= results
        print(subdir, ':', len(data_list))
    with open(jsonf, 'w') as json_file:
        json.dump(data_list, json_file, indent=4)



if __name__ == '__main__':
    # subidx = int(sys.argv[1])
    crop_save(patch_dir=f'{VLDATA_PROCESS}/segpath_cropped',
              jsonf = f'{VLDATA_PROCESS}/mix_segpath.json')

    # with open(f'{VLDATA_PROCESS}/mix_segpath.json', 'r') as json_file:
    #     data = json.load(json_file)
    # breakpoint()