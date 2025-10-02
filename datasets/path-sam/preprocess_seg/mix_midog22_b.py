
# https://midog.deepmicroscopy.org/download-dataset/
# https://midog2021.grand-challenge.org/
# https://drive.google.com/drive/folders/1YUMKNkXUtgaFM6jCHpZxIHPZx_CqE_qG
# https://drive.google.com/drive/folders/1P73g1xg8jw_JGLJaDFQDnxwQA7ROVykA
import os, glob, json
import numpy as np
import sys
from PIL import Image, ImageDraw
import sqlite3
from collections import defaultdict

sys.path.append('./')
from utils.cfg import VLDATA_RAW

"""
MIDOG22
Cases	Cancer type	Species	Other information (scanner, image source, etc.)
001.tiff to 150.tiff	breast cancer	human	UMC Utrecht, scanned with three different scanners (see MIDOG 2021)
151.tiff to 194.tiff	lung carcinoma	canine	VetMedUni Vienna, scanned with 3DHistech Pannoramic Scan II
195.tiff to 249.tiff	lymphoma	canine	VetMedUni Vienna, scanned with 3DHistech Pannoramic Scan II
250.tiff to 299.tiff	mast cell tumor	canine	FU Berlin, scanned with Aperio ScanScope CS2
300.tiff to 354.tiff	neuroendocrine tumor	human	UMC Utrecht, scanned with Hamamatsu XR
355.tiff to 405.tiff	melanoma	human	UMC Utrecht, scanned with Hamamatsu XR (no labels provided for this domain)
355-405 missing labels
"""

def read_sqlite(sqlitef):
    # Step 1: Connect to the SQLite database
    conn = sqlite3.connect(sqlitef)

    # Step 2: Create a cursor object
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print(cursor.fetchall())

    # Step 3: Execute an SQL query
    query = "PRAGMA table_info(Annotations_coordinates);"
    cursor.execute(query)

    # Step 4: Fetch the data
    rows = cursor.fetchall()

    # Step 5: Process or print the data
    for row in rows:
        print(row)

    # Step 6: Close the connection
    conn.close()


def read_image_bbox(jsonf, image_dir, process_img_idx=None, vis_dir=''):
    if vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
    with open(jsonf) as f:
        data = json.load(f)
    print(data['categories'])
    print(len(data['images']), len(data['annotations']))
    img_id_to_name = {d['id']:d['file_name'] for d in data['images']}
    img_id_to_wh = {d['id']:(d['width'], d['height']) for d in data['images']}
    # for idx, imgname in img_id_to_path.items():
    #     imgpath = f'{base_dir}/images/{imgname}'
    #     if not os.path.exists(imgpath):
    #         print(imgpath)
    # Assign bounding boxes to images
    image_2_bbox = defaultdict(lambda: defaultdict(list))
    for d in data['annotations']:
        image_name = img_id_to_name[d['image_id']]
        if process_img_idx is not None and image_name.split('.')[0] not in process_img_idx:
            continue
        bbox = d['bbox']
        category = str(d['category_id'])
        image_2_bbox[d['image_id']][category].append(bbox)
    # Visualize bbox on each image
    # [{'id': 1, 'name': 'mitotic figure'}, {'id': 2, 'name': 'not mitotic figure'}]
    for image_id, category_2_bbox in image_2_bbox.items():
        image_name = img_id_to_name[image_id]
        image_path = f'{image_dir}/{image_name}'
        print(image_path)
        image = Image.open(image_path).convert('RGB')
        wh = img_id_to_wh[image_id]
        assert wh == image.size
        width, height = image.size

        if vis_dir:
            draw = ImageDraw.Draw(image)

            for category, bboxes in category_2_bbox.items():
                color = 'red' if category == '1' else 'green'
                for bbox in bboxes:
                    x0, y0, x1, y1 = bbox
                    if not (x1<=width and y1<=height):
                        print('Warning', max(0, x1-width), max(0, y1-height))
                    else:
                        draw.rectangle(bbox, outline=color, width=3)
            image.save(f"{vis_dir}/{image_id}.jpg")
            print(f"Saved {image_id}.jpg: {len(category_2_bbox['1'])} mitoses, {len(category_2_bbox['2'])} non-mitoses")


if __name__ == '__main__':
    # read_sqlite(f'{VLDATA_RAW}/midog2/MIDOG2022_training.sqlite')
    read_image_bbox(jsonf=f'{VLDATA_RAW}/midog2/MIDOG2022_training.json',
               image_dir=f'{VLDATA_RAW}/midog2/images',
               process_img_idx=["{:03}".format(n) for n in np.arange(300, 354)],
               vis_dir = f'{VLDATA_RAW}/midog2/vis_neuroendocrine')
    read_image_bbox(jsonf=f'{VLDATA_RAW}/midog2/MIDOG2022_training.json',
               image_dir=f'{VLDATA_RAW}/midog2/images',
               process_img_idx=["{:03}".format(n) for n in np.arange(1, 150)],
               vis_dir = f'{VLDATA_RAW}/midog2/vis_breast')
    # read_image_bbox(jsonf=f'{VLDATA_RAW}/midog2/MIDOG2022_training.json',
    #            image_dir=f'{VLDATA_RAW}/midog2/images',
    #            process_img_idx=["{:03}".format(n) for n in np.arange(355, 405)],
    #            vis_dir = f'{VLDATA_RAW}/midog2_vis_melanoma')    # 355-405 missing labels
