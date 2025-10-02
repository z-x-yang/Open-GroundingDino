# https://zenodo.org/records/10560728

"""
This is the dataset from the publication "Cyto R-CNN and CytoNuke Dataset: Towards reliable whole-cell segmentation in bright-field histological images" by Raufeisen et al. (2024). It contains 6,683 annotations (3,991 nuclei and 2,607 whole cells) of head and neck squamous cell carcinoma cells in hematoxylin and eosin stained histological images. The annotations are in COCO format and distributed over 83 PNG images. Cyto R-CNN was trained on this dataset and compared with other state-of-the-art methods. The CytoNuke dataset is released under the CC BY-NC-SA 4.0 license.

The histological images are from the CPTAC dataset:
National Cancer Institute Clinical Proteomic Tumor Analysis Consortium (CPTAC). (2018). The Clinical Proteomic Tumor Analysis Consortium Head and Neck Squamous Cell Carcinoma Collection (CPTAC-HNSCC) (Version 15) [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2018.UW45NH81
"""


import os, glob, sys
from collections import defaultdict
import json
import numpy as np
import cv2
from PIL import Image
sys.path.append('./')
from utils.draw import draw_image
from utils.cfg import VLDATA_RAW, VLDATA_PROCESS

# Unknown type: cell and nucleus instances

if __name__ == '__main__':
    vis_dir = f'{VLDATA_RAW}/cytonuke/vis'
    os.makedirs(vis_dir, exist_ok=True)
    image_dir = f'{VLDATA_RAW}/cytonuke/images'
    annot = json.load(open(f'{VLDATA_RAW}/cytonuke/coco.json'))
    
    annot_image = annot['images']
    idx_to_image = {img['id']: img['file_name'] for img in annot_image}
    annot_cat = annot['categories']
    catid_to_category = {cat['id']: cat['name'] for cat in annot_cat}
    annot_inst = annot['annotations']
    image_to_cells = defaultdict(list)
    image_to_nucleus = defaultdict(list)
    for n, inst in enumerate(annot_inst):
        image_name = idx_to_image[inst['image_id']]
        image = np.array(Image.open(os.path.join(image_dir, image_name)).convert("RGB"))
        category = catid_to_category[inst['category_id']]
        bbox = inst['bbox']
        segs = inst['segmentation'] 

        for seg in segs:
            seg = np.array(seg).reshape((-1, 2))
            if category == 'CELL':
                image_to_cells[image_name].append(seg)
            elif category == 'NUCLEUS':
                image_to_nucleus[image_name].append(seg)
    for imagename, cells in image_to_cells.items():
        nucleus = image_to_nucleus[imagename]
        print(f"Image: {imagename}, Cells: {len(cells)}, Nuclei: {len(nucleus)}")
        # visualize contours
        image = np.array(Image.open(os.path.join(image_dir, imagename)))
        image = draw_image(image, cells,  color = (0,0,255)) #red
        image = draw_image(image, nucleus, color = (255,0,0)) #blue
        cv2.imwrite(f'{vis_dir}/{imagename}', image)
    # TODO: why some cells are not labeled in visualization