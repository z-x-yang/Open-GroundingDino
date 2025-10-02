# https://arxiv.org/pdf/2108.11195

import scipy.io as sio
import numpy as np
import os, glob
import json
import cv2
from PIL import Image
import sys
sys.path.append('./')

from utils.cfg import VLDATA_RAW, VLDATA_PROCESS
from preprocess.tile_utils import crop_img_mask_to_caption
from utils.draw import draw_image, COLORS

_VIS_DIR = f'{VLDATA_PROCESS}/visualization/lizard'
os.makedirs(_VIS_DIR, exist_ok=True)

_RAW_DIR = f'{VLDATA_RAW}/Lizard'
_GT_LABEL = {
    1: "Neutrophil",
    2: "Epithelial",
    3: "Lymphocyte",
    4: "Plasma",
    5: "Eosinophil",
    6: "Connective tissue"}

# from read_label.py
def read_label_mat(mat_file):
    class_bboxs = {gt: [] for gt in _GT_LABEL.keys()}
    class_centroids = {gt: [] for gt in _GT_LABEL.keys()}

    """Loading the labels within the Lizard dataset."""
    label = sio.loadmat(mat_file) #! This filename is a placeholder!

    # Load the instance segmentation map.
    # This map is of type int32 and contains values from 0 to N, where 0 is background
    # and N is the number of nuclei. 
    # Shape: (H, W) where H and W is the height and width of the image.
    inst_map = label['inst_map'] 

    # Load the index array. This determines the mapping between the nuclei in the instance map and the
    # corresponing provided categories, bounding boxes and centroids.
    nuclei_id = label['id'] # shape (N, 1), where N is the number of nuclei.

    # Load the nuclear categories / classes. 
    # Shape: (N, 1), where N is the number of nuclei.
    classes = label['class']

    # Load the bounding boxes.
    # Shape: (N, 4), where N is the number of nuclei.
    # For each row in the array, the ordering of coordinates is (y1, y2, x1, x2). 
    bboxs = label['bbox'] 

    # Load the centroids.
    # Shape: (N, 2), where N is the number of nuclei.
    # For each row in the array, the ordering of coordinates is (x, y).
    centroids = label['centroid'] 

    # Matching each nucleus with its corresponding class, bbox and centroid:

    # Get the unique values in the instance map - each value corresponds to a single nucleus.
    unique_values = np.unique(inst_map).tolist()[1:] # remove 0

    # Convert nuclei_id to list.
    nuclei_id = np.squeeze(nuclei_id).tolist()
    for value in unique_values:
        # Get the position of the corresponding value
        idx = nuclei_id.index(value)
        
        class_ = classes[idx][0]
        bbox = bboxs[idx]
        centroid = centroids[idx]

        class_bboxs[class_].append(bbox)
        class_centroids[class_].append(centroid)
    return class_bboxs, class_centroids

def get_statistics(class_bbox):
    cell_cnt = {gt: 0 for gt in _GT_LABEL.keys()}
    cell_occupy = {gt: 0 for gt in _GT_LABEL.keys()}

    cell_cnt[class_] += 1
    cell_occupy[class_] += (bbox[1]-bbox[0])*(bbox[3]-bbox[2])
    
def get_images_masks():
    image_dir1 = f'{_RAW_DIR}/lizard_images1/Lizard_Images1'
    image_dir2 = f'{_RAW_DIR}/lizard_images2/Lizard_Images2'
    image_paths = glob.glob(f'{image_dir1}/*.png') + glob.glob(f'{image_dir2}/*.png')
    image_names = [p.split('/')[-1].split('.')[0] for p in image_paths]
    label_mat_dir = f'{_RAW_DIR}/lizard_labels/Lizard_Labels/Labels'
    # for labelpath in glob.glob(f'{label_mat_dir}/*.mat'):
    label_paths = [f'{label_mat_dir}/{iname}.mat' for iname in image_names]
    assert all([os.path.exists(lpath) for lpath in label_paths])
    print(len(image_paths))
    return image_names, image_paths, label_paths

def update_mask(height_width, class_bboxs):
    "BBox: (y1, y2, x1, x2)"
    mask = np.zeros(height_width)
    for class_, bboxs in class_bboxs.items():
        for bbox in bboxs:
            mask[bbox[0]:bbox[1], bbox[2]:bbox[3]] = class_
    return mask


def visualize_cells(image, class_bboxs, imgname):
    "BBox: (y1, y2, x1, x2)"
    for idx, (class_, bboxs) in enumerate(class_bboxs.items()):
        for bbox in bboxs:
            y1, y2, x1, x2 = bbox
            image = cv2.rectangle(image, (x1,y1), (x2,y2), COLORS[idx], 1)
    cv2.imwrite(f'{_VIS_DIR}/{imgname}',  cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    # 238 * (500, 500)~(1500, 1500), BBox and centroid
    crop_dir = f'{VLDATA_PROCESS}/lizard_cropped'
    os.makedirs(crop_dir, exist_ok=True)
    datalist = []
    image_names, image_paths, label_paths = get_images_masks()
    for ipath, lpath in zip(image_paths, label_paths):
        img = Image.open(ipath)
        class_bboxs, class_centroids = read_label_mat(lpath)
        # visualize_cells(np.array(img), class_bboxs, imgname=ipath.split('/')[-1])
        # continue
        width, height = img.size
        mask = update_mask((height, width), class_bboxs)
        # Optimize crop size
        standard_hw = 336
        height, width = mask.shape[:2]
        num_width, num_height = width//standard_hw+1, height//standard_hw+1
        # num_width, num_height = max(num_width, 1), max(1, num_height)
        if max(num_width, num_height)<=2:
            tile_hw = min(height//num_height, width//num_width)
        else:
            tile_hw = standard_hw
        num_tile = (height//tile_hw)*(width//tile_hw)
        print(height, width, tile_hw, '#',num_tile)
        wsi_mark = ipath.split('/')[-2] +'-'+ ipath.split('/')[-1][:-4]
        # 
        results = crop_img_mask_to_caption(whole_img=np.array(img),
                            whole_mask=mask[:,:,None],
                            out_dir=crop_dir,
                            wsi_mark=wsi_mark,
                            label_def_dict=_GT_LABEL,
                            tile_hw=tile_hw,
                            label_centroids=class_centroids)
        datalist += results

    print(len(datalist))
    with open(f"{VLDATA_PROCESS}/colon_lizard.json", "w") as file:
        json.dump(datalist, file, indent=4)
