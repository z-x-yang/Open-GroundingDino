# https://sites.google.com/view/panoptils/
# images are from TCGA

import numpy as np
import os, json
import glob
import pandas as pd
from tqdm import tqdm
from PIL import Image
import numpy as np
from scipy.ndimage import label
from collections import defaultdict
import sys
sys.path.append('./')
from utils.cfg import VLDATA_RAW, VLDATA_PROCESS
from preprocess.tile_utils import crop_img_mask_to_caption

"""
Size (1024,1024)
> rgbs/: These are the RGB images in .png format
> masks/: These are the corresponding segmentation masks. They have a three-channel .png format. 
    > First channel: is the region semantic segmentation mask. 
        0:	Exclude
        1: 	Cancerous epithelium
        2:	Stroma
        3:	TILs
        4:	Normal epithelium
        5:	Junk/Debris
        6:	Blood
        7:	Other
        8:	Whitespace/Empty
        Suggested grouping:
        Epithelium = 1 + 4
    > Second channel: is the nucleus semantic segmentation mask. 
        0:	Exclude
        1:	Cancer nucleus
        2:	Stromal nucleus
        3:	Large stromal nucleus
        4:	Lymphocyte nucleus 
        5:	Plasma cell / large TIL nucleus
        6:	Normal epithelial nucleus
        7:	Other nucleus
        8:	Unknown/Ambiguous nucleus
        9:	Background (non-nuclear material)
        Suggested grouping:
        Epithelial = 1 + 6
        Stromal = 2 + 3
        TILs = 4 + 5
    > Third channel: is a binary mask of nuclear boundary edges.
> csv/: These are the classification labels and segmentation boundary coordinates. 
> vis/: These are visualizations of the segmentation masks, provided for convenience.
"""

_TISSUE_LABEL_DEF = {
    0:	"Exclude",
    1: 	"Cancerous epithelium",
    2:	"Stroma",
    3:	"TILs",
    4:	"Normal epithelium",
    5:	"Junk/Debris",
    6:	"Blood",
    7:	"Other",
    8:	"Whitespace/Empty"
}
_TISSUE_REMOVE_LABEL = [0, 7]
_NUCLEUS_LABEL_DEF = {
    0:	"Exclude",
    1:	"Cancer nucleus",
    2:	"Stromal nucleus",
    3:	"Large stromal nucleus",
    4:	"Lymphocyte nucleus",
    5:	"Plasma cell / large TIL nucleus",
    6:	"Normal epithelial nucleus",
    7:	"Other nucleus",
    8:	"Unknown/Ambiguous nucleus",
    9:	"Background (non-nuclear material)"
}
_NUCLEUS_REMOVE_LABEL = [0, 9]

_RAW_DIR = f'{VLDATA_RAW}/PanopTILs_processed'
_IMG_SIZE = (1024, 1024)

def read_images_masks(base_dirs):
    imagefs, maskfs = [], []
    for base in base_dirs:
        imagefs += glob.glob(f'{base}/rgbs/*.png')
    print(len(imagefs))
    for imagef in (imagefs):
        maskf = imagef.replace('/rgbs/', '/masks/')
        # mask = np.array(Image.open(maskf))
        # mask_tissue, mask_nuclus = mask[:,:,0], mask[:,:,1]
        # image = np.array(Image.open(imagef))
        # assert set(np.unique(mask_tissue)).issubset(set(_TISSUE_LABEL_DEF.keys()))
        # assert set(np.unique(mask_nuclus)).issubset(set(_NUCLEUS_LABEL_DEF.keys()))
        # assert image.shape[:2] == _IMG_SIZE == mask.shape[:2]
        maskfs.append(maskf)
    return imagefs, maskfs

def get_nuclus_centriod_xy(nuclus_label_def, mask):
    from scipy.ndimage import label
    label_centroids = defaultdict(list)
    for k, v in nuclus_label_def.items():
        labeled_array, num_features = label(mask==k)
        # Calculate centroids
        for instance in range(1, num_features + 1):  # Skip background (label 0)
            coords = np.column_stack(np.where(labeled_array == instance))
            centroid_xy = coords.mean(axis=0)
            label_centroids[v].append(centroid_xy)
    return label_centroids

def count_nuclus_in_bbox(label_centroids, x0, y0, x1, y1 ):
    count = {k:0 for k in label_centroids.keys()}
    for k, clist in label_centroids.items():
        for cx, cy in clist:
            if x0<=cx<=x1 and y0<=cy<=y1:
                count[k] += 1
    return count

def crop_save(imagefs, maskfs, crop_dir, jsonf):
    os.makedirs(crop_dir, exist_ok=True)
    datalist = []

    tissue_label_def = {k:v for k, v in _TISSUE_LABEL_DEF.items() if not k in _TISSUE_REMOVE_LABEL}
    nuclus_label_def = {k:v for k, v in _NUCLEUS_LABEL_DEF.items() if not k in _NUCLEUS_REMOVE_LABEL}
    for imagef, maskf in tqdm(zip(imagefs, maskfs)):
        img = np.array(Image.open(imagef))
        mask = np.array(Image.open(maskf))
        mask_tissue, mask_nuclus = mask[:,:,0], mask[:,:,1]
        nuclus_label_centroids = get_nuclus_centriod_xy(nuclus_label_def, mask_nuclus)

        results = crop_img_mask_to_caption(whole_img=img,
                                            whole_mask=mask_tissue[:,:,None],
                                            out_dir=crop_dir,
                                            wsi_mark=imagef.split('/')[-1][:-4],
                                            label_def_dict=tissue_label_def,
                                            tile_hw=341) #1024%341=1
        total_count = {k: 0 for k in nuclus_label_centroids.keys()}
        for data in results:
            imgname = data['image'].split('/')[-1][:-4]
            x0, y0 = imgname.split('_')[-2:]
            x0, y0 = int(x0), int(y0)
            x1, y1 = x0+341, y0+341
            count_dict = count_nuclus_in_bbox(nuclus_label_centroids, x0, y0, x1, y1)
            data.update({'count':count_dict})
            total_count = {k: total_count[k]+v for k, v in count_dict.items()}
            datalist += [data]
            # print(count_dict)
        # print(results)
        for k,v in nuclus_label_centroids.items():
            if not total_count[k] == len(v):
                print(f'{total_count[k]} vs {len(v)}')
    print(len(datalist))
    with open(jsonf, "w") as file:
        json.dump(datalist, file, indent=4) 

if __name__ == '__main__':
    dir1 = f'{_RAW_DIR}/bootstrap_nuclei/tcga'
    dir2 =  f'{_RAW_DIR}/manual_nuclei'
    imagefs, maskfs = read_images_masks([dir1, dir2])

    jsonf = f'{VLDATA_PROCESS}/breast_panoptils.json'
    if not os.path.exists(jsonf):
        crop_save(imagefs, maskfs, f'{VLDATA_PROCESS}/panoptils_cropped', jsonf)
    else:
        with open(jsonf) as f:
            data = json.load(f)
        breakpoint()