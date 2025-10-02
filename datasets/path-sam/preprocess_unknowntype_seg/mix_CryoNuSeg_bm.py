# https://www.kaggle.com/datasets/ipateam/segmentation-of-nuclei-in-cryosectioned-he-images
# https://www.sciencedirect.com/science/article/pii/S0010482521001438

import json
import os, glob, sys
import numpy as np
from PIL import Image
sys.path.append('./')
from utils.cfg import VLDATA_RAW

if __name__ == '__main__':
    image_dir = f'{VLDATA_RAW}/CryoNuSeg/tissue_images'
    label_dir = f'{VLDATA_RAW}/CryoNuSeg/Annotator_1_manual_up/Annotator_1_manual_up/label_masks'

    for imagepath in glob.glob(f'{image_dir}/*.tif'):
        name = os.path.basename(imagepath)
        labelpath = f'{label_dir}/{name}'
        img = np.array(Image.open(imagepath))
        label = np.array(Image.open(labelpath))
        height, width = img.shape[0], img.shape[1]
        assert img.shape[0:2] == label.shape[0:2], f"Image and label shape mismatch: {img.shape} vs {label.shape}"
        # check label values
        labels = np.unique(label)
        print(f"Unique label values in {labelpath}: {len(labels)}")
