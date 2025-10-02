# https://tiger.grand-challenge.org/Data/


import numpy as np
import pandas as pd
from PIL import Image

import os, sys, json
import numpy as np
from PIL import Image
import pandas as pd
import os
import tifffile as tiff
import json

Image.MAX_IMAGE_PIXELS = None


sys.path.append('./')
from utils.cfg import VLDATA_RAW, VLDATA_PROCESS
from preprocess.tile_utils import crop_img_mask_to_caption


_RAW_DIR = f'{VLDATA_RAW}/TIGER'

"""
	|_wsibulk/
	|	|__annotations-tumor-bulk/				* manual annotations of "tumor bulk" regions (see https://tiger.grand-challenge.org/Data/ for details)
	|	|	|___masks/							* annotations in multiresolution TIF format
	|	|	|___xmls/							* annotations in XML format
	|	|__images/								* whole-slide images
	|	|__tissue-masks/						* tissue-background masks
	|
	|_wsirois/
	|	|__roi-level-annotations/				* manual annotations of tissue and cells on cropped reegions of interest
	|	|	|___tissue-bcss/					* manual annotations of tissue only in larger ROIs adapted from the BCSS project
	|	|	|	|____images/					* images in PNG format of ROIs extracted from whole-slide images (TCGA data only)
	|	|	|	|____masks/						* manual annotations of tissue in ROIs
	|	|	|___tissue-cells/					* manual annotations of tissue and cells ROIs adapted from the NuCLS+BCSS project and from RUMC and JB data
	|	|		|____images/					* images in PNG format of ROIs extracted from whole-slide images (TCGA + RUMC + JB)
	|	|		|____masks/						* manual annotations of tissue in ROIs
	|	|		|____tiger-coco.json			* manual annotations of cells in ROIs as bounding boxes in COCO format
	|	|__wsi-level-annotations/				* manual annotations of tissue and cells on whole-slide images
	|		|___annotations-tissue-bcss-masks/	* manual annotations (in TIF format) of tissue only adapted from the BCSS project
	|		|___annotations-tissue-bcss-xmls/	* manual annotations (in XML format) of tissue only adapted from the BCSS project
	|		|___annotations-tissue-cells-masks/	* manual annotations (in TIF format) of tissue and cells from RUMC and JB + data adapted from the NuCLS+BCSS project 
	|		|___annotations-tissue-cells-xmls/	* manual annotations (in XML format) of tissue and cells from RUMC and JB + data adapted from the NuCLS+BCSS project 
	|		|___images/							* whole-slide images
	|		|___tissue-masks/					* tissue-background masks
	|
	|_wsitils/									
	|	|__images/								* whole-slide images
	|	|__tissue-masks/						* tissue-background masks
	|	|__tiger-tils-scores-wsitils.csv		* CSV file containing TILs scores for each WSI
	|
	|_data-structure.txt						* this file
"""


# invasive tumor (label=1): this class contains regions of the invasive tumor, including several morphological subtypes, such as invasive ductal carcinoma and invasive lobular carcinoma;
# tumor-associated stroma (label=2): this class contains regions of stroma (i.e., connective tissue) that are associated with the tumor. This means stromal regions contained within the main bulk of the tumor and in its close surrounding; in some cases, the tumor-associated stroma might resemble the "healthy" stroma, typically found outside of the tumor bulk; 
# in-situ tumor (label=3): this class contains regions of in-situ malignant lesions, such as ductal carcinoma in situ (DCIS) or lobular carcinoma in situ (LCIS).
# healthy glands (label=4): this class contains regions of glands with healthy epithelial cells;
# necrosis not in-situ (label=5): this class contains regions of necrotic tissue that are not part of in-situ tumor; for example, ductal carcinoma in situ (DCIS) often presents a typical necrotic pattern, which can be considered as part of the lesion itself, such a necrotic region is not annotated as "necrosis" but as "in-situ tumor";
# inflamed stroma (label=6): this class contains tumor-associated stroma that has a high density of lymphocytes (i.e., it is "inflamed"). When it comes to assessing the TILs, inflamed stroma and tumor-associated stroma can be considered together, but were annotated separately to take into account for differences in their visual patterns;
# rest (label=7): this class contains regions of several tissue compartments that are not specifically annotated in the other categories; examples are healthy stroma, erythrocytes, adipose tissue, skin, nipple, etc.
_GT_LABEL = {
    1: "invasive tumor",
    2: "tumor-associated stroma",
    3: "in-situ tumor",
    4: "healthy glands",
    5: "necrosis not in-situ",
    6: "inflamed stroma",
    7: "rest"
}

def read_wsi_tissuemask_TILscore():
    # TIL: tumor-infiltrating lymphocytes
    # WSI level TIL-score and comment
    # WSI tissue mask
    tilf = f'{_RAW_DIR}/wsitils/tiger-til-scores-wsitils.csv' 
    imgdir = f'{_RAW_DIR}/wsitils/images'
    maskdir = f'{_RAW_DIR}/wsitils/tissue-masks'
    for imgf in os.listdir(imgdir):
        maskf = imgf.replace('.tif', '_tissue.tif')
        img = np.array(Image.open(f'{imgdir}/{imgf}'))

        # mask = Image.open(f'{maskdir}/{maskf}')
        mask = tiff.imread(f'{maskdir}/{maskf}')
        mask = np.array(mask)
        print(img.shape, mask.shape)
        print(np.unique(mask[::100, ::100]))

def read_wsirois():
    imgdir = f'{_RAW_DIR}/wsirois/images'
    pass

def read_wsibulks_tissuemask():
    "we made sure that all cancer cells belonging to the invasive part of the tumor are confined within the manually annotated regions."
    imgdir = f'{_RAW_DIR}/wsibulk/images'
    # maskdir = f'{_RAW_DIR}/wsibulk/tissue-masks'
    maskdir = f'{_RAW_DIR}/wsibulk/annotations-tumor-bulk/masks'
    xmldir = f'{_RAW_DIR}/wsibulk/annotations-tumor-bulk/xmls' #contour XML
    for imgf in os.listdir(imgdir):
        maskf = imgf
        xmlf = imgf.replace('.tif', '.xml')
        img = np.array(Image.open(f'{imgdir}/{imgf}'))
        mask = tiff.imread(f'{maskdir}/{maskf}')
        mask = np.array(mask)
        print(img.shape)
        print(np.unique(mask[::100, ::100]))#[0,1]

#Disregard relabeling of NuCLS data
# 1:lymphocytes and plasma cells; 0: exclude

def read_wsirois_cell(): #1879 pairs
    imgfs, maskfs = [], []
    # manual annotations of tissue and cells ROIs adapted from the NuCLS+BCSS project and from RUMC and JB data
    imgdir = f'{_RAW_DIR}/wsirois/roi-level-annotations/tissue-cells/images'
    maskdir = f'{_RAW_DIR}/wsirois/roi-level-annotations/tissue-cells/masks'
    
    #Label.v1: cells in ROIs label
    jsonf = f'{_RAW_DIR}/wsirois/roi-level-annotations/tissue-cells/tiger-coco.json'
    with open(jsonf, 'r') as file:
        data = json.load(file)
    print(data['categories']) # [{'id': 1, 'name': 'lymphocytes and plasma cells'}]
    
    #Label.v2: tissue label
    labels = set()
    for imgf in os.listdir(imgdir):
        try:
            maskf = imgf
            img = np.array(Image.open(f'{imgdir}/{imgf}'))
            mask = np.array(Image.open(f'{maskdir}/{maskf}'))
            assert img.shape[:2] == mask.shape
            for l in np.unique(mask):
                labels.add(l)
            imgfs.append(f'{imgdir}/{imgf}')
            maskfs.append(f'{maskdir}/{maskf}')
        except:
            print(imgf)
    print(labels)
    print(len(imgfs))
    return imgfs, maskfs

def read_wsirois_tissue(): #151 paired from TCGA
    imgfs, maskfs = [], []
    imgdir = f'{_RAW_DIR}/wsirois/roi-level-annotations/tissue-bcss/images'
    maskdir = f'{_RAW_DIR}/wsirois/roi-level-annotations/tissue-bcss/masks'
    labels = set()
    for imgf in os.listdir(imgdir):
        try:
            maskf = imgf
            img = np.array(Image.open(f'{imgdir}/{imgf}'))
            mask = np.array(Image.open(f'{maskdir}/{maskf}'))
            assert img.shape[:2] == mask.shape
            for l in np.unique(mask):
                labels.add(l)
            imgfs.append(f'{imgdir}/{imgf}')
            maskfs.append(f'{maskdir}/{maskf}')
        except:
            print(imgf)
    print(labels) #{0, 1, 2, 3, 4, 5, 6, 7}
    print(len(imgfs))
    return imgfs, maskfs

def cutomize_best_patchsize(height, width):
    # Optimize crop size
    standard_hw = 336
    
    num_width, num_height = width//standard_hw+1, height//standard_hw+1
    # num_width, num_height = max(num_width, 1), max(1, num_height)
    if max(num_width, num_height)<=2:
        tile_hw = min(height//num_height, width//num_width)
    else:
        tile_hw = standard_hw
    num_tile = (height//tile_hw)*(width//tile_hw)
    print(height, width, tile_hw, '#',num_tile)
    return tile_hw, num_tile

def crop_save_patches(imagefs, maskfs, crop_dir, jsonf):
    os.makedirs(crop_dir, exist_ok=True)

    datalist = []
    for imagef, maskf in zip(imagefs, maskfs):
        img = np.array(Image.open(imagef))
        mask = np.array(Image.open(maskf))
        tile_hw, num_tile = cutomize_best_patchsize(mask.shape[0], mask.shape[1])
        wsi_mark = imagef.split('/')[-1][:-4]
        results = crop_img_mask_to_caption(whole_img=np.array(img),
                            whole_mask=mask[:,:,None],
                            out_dir=crop_dir,
                            wsi_mark=wsi_mark,
                            label_def_dict=_GT_LABEL,
                            tile_hw=tile_hw)
        datalist += results
    print(len(datalist))
    with open(jsonf, 'w') as json_file:
        json.dump(datalist, json_file, indent=4)

if __name__ == '__main__':
    # read_wsi_tissuemask_TILscore()
    # read_wsibulks_tissuemask()

    # 1. Read large patches
    # cell: (100, 100)~（2000,2000）
    # tissue: (1000, 1000)~（2000,2000）
    imagefs1, maskfs1 = read_wsirois_cell()
    imagefs2, maskfs2 = read_wsirois_tissue()

    # 2. Crop into 336 standard patches
    crop_save_patches(imagefs1+imagefs2, maskfs1+maskfs2,
                      crop_dir=f'{VLDATA_PROCESS}/tiger_cropped',
                      jsonf=f'{VLDATA_PROCESS}/breast_tiger.json')
