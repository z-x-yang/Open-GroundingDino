# https://bcsegmentation.grand-challenge.org/

import numpy as np
import os, json
from PIL import Image
from tqdm import tqdm

import sys
sys.path.append('./')
from utils.cfg import VLDATA_RAW, VLDATA_PROCESS
from preprocess.tile_utils import crop_img_mask_to_caption

_RAW_DIR = f'{VLDATA_RAW}/BCSS/BCSS'
_IMG_DIR = f'{_RAW_DIR}/images'
_MASK_DIR = f'{_RAW_DIR}/masks'



_LABEL_DEF = {
    0: "outside_roi", # remove
    1: "tumor",  
    2: "stroma",
    3: "lymphocytic_infiltrate",
    4: "necrosis_or_debris",
    5: "glandular_secretions",
    6: "blood",
    7: "exclude", # remove
    8: "metaplasia_NOS",
    9: "fat",
    10: "plasma_cells",
    11: "other_immune_infiltrate",
    12: "mucoid_material",
    13: "normal_acinus_or_duct",
    14: "lymphatics",
    15: "undetermined", # remove
    16: "nerve",
    17: "skin_adnexa",
    18: "blood_vessel",
    19: "angioinvasion",
    20: "dcis",
    21: "other", # remove
}
_REMOVE_LABELS = [0, 7, 15, 21]
_SAVED_LABEL_DEF = {k: v for k, v in _LABEL_DEF.items() if k not in _REMOVE_LABELS}

if __name__ == '__main__':
    jsonf = f'{VLDATA_PROCESS}/breast_bcss.json'
    if not os.path.exists(jsonf):
        crop_dir = f'{VLDATA_PROCESS}/bcss_cropped'
        os.makedirs(crop_dir, exist_ok=True)
        datalist = []
        # Image/mask regions: 1000~8000
        for imgname in tqdm(os.listdir(_IMG_DIR)):
            imgf = f'{_IMG_DIR}/{imgname}'
            maskf = f'{_MASK_DIR}/{imgname}'
            img = np.array(Image.open(imgf))
            mask = np.array(Image.open(maskf))
            height, width = img.shape[:2]
            assert set(np.unique(mask)).issubset(set(_LABEL_DEF.keys()))
            print(height, width)
            results = crop_img_mask_to_caption(whole_img=img,
                                                whole_mask=mask[:,:,None],
                                                out_dir=crop_dir,
                                                wsi_mark=imgname[:-4],
                                                label_def_dict=_SAVED_LABEL_DEF,
                                                tile_hw=336)
            results = [r for r in results if r]
            datalist += results
        print(len(datalist))
        with open(jsonf, 'w') as json_file:
            json.dump(datalist, json_file, indent=4)
    with open(jsonf) as json_file:
        data = json.load(json_file)
    breakpoint()