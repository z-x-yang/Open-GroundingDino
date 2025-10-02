# origin: https://github.com/QuIIL/Sonnet/blob/main/extract_patches.py
# download from: https://github.com/hvcl/DiffMix
# data explaination: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9709151


import os, glob, sys
from collections import defaultdict
import json
import numpy as np
import cv2
from scipy.io import loadmat
from PIL import Image
sys.path.append('./')
from utils.draw import draw_image
from utils.cfg import VLDATA_RAW, VLDATA_PROCESS

# nuclei segmentation


if __name__ == '__main__':
    image_dir = [f'{VLDATA_RAW}/glysac_dataset/Train/Images']
    label_dir = [f'{VLDATA_RAW}/glysac_dataset/Train/Labels']

    labels  = set()
    for _idir, _ldir in zip(image_dir, label_dir):
        for imagepath in glob.glob(f'{_idir}/*.png'):
            imgname = os.path.basename(imagepath)
            labelpath = os.path.join(_ldir, imgname.replace('.png', '.mat'))
            img = Image.open(imagepath)
            width, height = img.size
            label = loadmat(labelpath)
            inst_map = label['inst_map']
            inst_type = label['inst_type']
            ann_type = label['type_map']
            anns = np.unique(ann_type)
            for ann in anns:
                labels.add(ann)
            # ann_type[(ann_type == 1) | (ann_type == 2) | (ann_type == 9) | (ann_type == 10)] = 1
            # ann_type[(ann_type == 4) | (ann_type == 5) | (ann_type == 6) | (ann_type == 7)] = 2
            # ann_type[(ann_type == 8) | (ann_type == 3)] = 3
    print(labels) #{0, 1, 2, 3, 4, 5, 6, 7, 8}
    #TODO: check label type
    """
    various types of nuclei
    exist, including lymphocytes, cancerous epithelial and normal
    epithelial nuclei, stromal nuclei, endothelial nuclei, and etc
    final 3: epithelial, lymphocytes, miscellaneous
    """

"""

Tan N. N. Doan is with the Department of Computer Science and
Engineering, Sejong University, Gwangjin-gu, Seoul 05006, South Korea
(e-mail: tandoan.hcmut@gmail.com).
Boram Song and Kyungeun Kim are with the Department of
Pathology, Kangbuk Samsung Hospital, Sungkyunkwan University
School of Medicine, Suwon, Seoul 05505, South Korea (e-mail: boram62.song@samsung.com; ke23.kim@samsung.com).
Trinh T. L. Vuong and Jin T. Kwak are with the School of Electrical
Engineering, Korea University, Seongbuk-gu, Seoul 02841, South Korea
(e-mail: vuongthiletrinh1995@gmail.com; jkwak@korea.ac.kr).
Digital Object Identifier 10.1109/JBHI.2022.3149936

"""