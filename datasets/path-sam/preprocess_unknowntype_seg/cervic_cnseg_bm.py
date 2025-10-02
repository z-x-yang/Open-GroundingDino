# https://www.kaggle.com/datasets/zhaojing0522/cervical-nucleus-segmentation?resource=download
# https://www.sciencedirect.com/science/article/pii/S016926072300398X


# We annotate 85,882 nuclei in these images. The background of these images is complex (shown in Fig. 3), including keratinized cells, microbially infected cells, dark spots, dense glandular cell clusters, atrophic cells, neutrophils, abnormal cell clusters, and cells with unclear boundaries

import json
import os, glob, sys
import numpy as np
from PIL import Image
sys.path.append('./')
from utils.cfg import VLDATA_RAW

if __name__ == '__main__':
    datadir = f'{VLDATA_RAW}/cnseg'
    label_subfolders = [f'clusteredCell/{s}_json' for s in ['difficult', 'normal', 'sample', 'test-difficult',
                                                            'test-normal', 'test-sample']]
    image_subfolders = [f'clusteredCell/{s}' for s in ['difficult', 'normal', 'sample', 'test-difficult',
                                                            'test-normal', 'test-sample']]
    label_subfolders += ['PatchSeg/train-labels', 'PatchSeg/test-labels', 'TargetA/test_json', 'TargetB/test-labels']
    image_subfolders += ['PatchSeg/train-images', 'PatchSeg/test-images', 'TargetA/test_images', 'TargetB/test-images']
    labels = set()
    for image_f, label_f in zip(image_subfolders, label_subfolders):
        imgpaths = glob.glob(f'{datadir}/{image_f}/*.png') + glob.glob(f'{datadir}/{image_f}/*.jpg')
        print(image_f, len(imgpaths))
        for imgpath in imgpaths:
            name = os.path.basename(imgpath).replace('.png', '').replace('.jpg', '')
            labelpath = f'{datadir}/{label_f}/{name}.json'
            label = json.load(open(labelpath))
            img = np.array(Image.open(imgpath))
            height, width = img.shape[0], img.shape[1]
            for inst in label['shapes']:
                label = inst['label']
                labels.add(label)
                contours = np.array(inst['points'], dtype=np.int32)
                pts_x, pts_y = contours[:, 0], contours[:, 1]
                if len(contours)<5:
                    print(f'{len(contours)} points, label: {label}')
                assert np.min(pts_x) >= 0
                assert np.min(pts_y) >= 0
                if not np.max(pts_x) <= width:
                    f"Point {np.max(pts_x)} is out of image width {width} x {height}"
                if not np.max(pts_y) <= height:
                    print(f"Point {np.max(pts_y)} is out of image height {width} x {height}")
    print(labels)# {'nuc', 'nn', 'nucleus', 'b', 'm'}
    # TODO: check number of points