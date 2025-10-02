# https://zenodo.org/records/15050523
# https://puma.grand-challenge.org/puma/

import os, glob, json
import numpy as np
import cv2
import sys
from PIL import Image, ImageDraw
import sqlite3
from collections import defaultdict
from shapely.geometry import shape
from shapely.geometry import Point, Polygon
sys.path.append('./')
from utils.cfg import VLDATA_RAW


def check_instance_countour(inst_countour):
    assert inst_countour.shape[1]==2, inst_countour.shape
    x_min, x_max = np.min(inst_countour[:,0]), np.max(inst_countour[:,0])-1
    y_min, y_max = np.min(inst_countour[:,1]), np.max(inst_countour[:,1])-1
    assert 0<=x_min<width and 0<=x_max<width, (x_min, x_max, width)
    assert 0<=y_min<height and 0<=y_max<height, (y_min, y_max, height)

def process_polygon(geom, image, vis_path=''):
    poly = Polygon(geom.exterior.coords)
    image = np.array(image)
    # interior: we can safely disregard holes, only use exterior to generate instance mask
    for hole in geom.interiors:
        hole_pts = list(hole.coords)
        inside = [Point(p).within(poly) for p in hole_pts]
        print(len(hole_pts), inside)
        # draw interior 
        pts = np.array(hole_pts).reshape((-1, 1, 2)).astype(np.int32)
        image = cv2.drawContours(image, [pts], -1,  (255,0,255), 2, cv2.LINE_8)
    if len(geom.interiors) and vis_path:
        # draw exterior 
        pts = np.array(geom.exterior.coords).reshape((-1, 1, 2)).astype(np.int32)
        image = cv2.drawContours(image, [pts], -1,  (0,0,255), 2, cv2.LINE_8)
        cv2.imwrite(vis_path, image)
    return len(geom.interiors)


if __name__ == '__main__':
    image_dir = f'{VLDATA_RAW}/puma/01_training_dataset_tif_ROIs'
    label_dir = f'{VLDATA_RAW}/puma/01_training_dataset_geojson_nuclei'
    vis_dir = f'{VLDATA_RAW}/puma/vis_nuclei'
    os.makedirs(vis_dir, exist_ok=True)

    total_cnt = defaultdict(int)
    for labelpath in glob.glob(f'{label_dir}/*.geojson'):
        image_id = os.path.basename(labelpath).split('_nuclei.geojson')[0]
        try:
            imagepath = os.path.join(image_dir, f'{image_id}.tif')
            img = Image.open(imagepath).convert('RGB')
        except:
            imagepath = os.path.join(image_dir, f'{image_id}.tiff')
            img = Image.open(imagepath).convert('RGB')
        width, height = img.size
        # print(image_id, img.size)
        with open(labelpath, 'r') as f:
            annot = json.load(f)
        # print(annot.keys(), annot['features'][0].keys())
        
        for feat in annot['features']:
            cls_type = feat['properties']['classification']['name']
            total_cnt[cls_type] += 1
            countour_coords = feat['geometry']['coordinates']
            geom = shape(feat['geometry']) 
            # print(geom.area, geom.bounds)
            if geom.geom_type == 'Polygon':
                process_polygon(geom, img, vis_path=f'{vis_dir}/{image_id}_poly.png')
                
            elif geom.geom_type == 'MultiPolygon':
                for i, subgeom in enumerate(geom.geoms):
                    process_polygon(subgeom, img, vis_path=f'{vis_dir}/{image_id}_poly.png')

    print(total_cnt)
    # {'nuclei_tumor': 57419, 'nuclei_apoptosis': 1850, 'nuclei_lymphocyte': 21643, 'nuclei_endothelium': 1701, 'nuclei_plasma_cell': 520, 'nuclei_stroma': 3856, 'nuclei_histiocyte': 7168, 'nuclei_melanophage': 695, 'nuclei_neutrophil': 366, 'nuclei_epithelium': 2211}