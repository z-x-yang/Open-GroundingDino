import os
import glob
from typing import List, Dict, Tuple

import os
import glob
import json
import pandas as pd
from PIL import Image


BBox = Tuple[int, int, int, int]


def read_ihc_tlymphoctype(root_dir: str) -> List[Dict]:
    """
    Read IHC T-lymphocyte dataset where each .tif has a paired .csv or _consensus.csv
    columns with XMin, YMin, XMax, YMax and Cell Type.

    Returns a list of dict per image:
      {
        'image_path': str,
        'width': int, 'height': int,
        'instances': [{'bbox': (x0,y0,x1,y1), 'label': str}, ...]
      }
    """
    records: List[Dict] = []
    for img_path in glob.glob(os.path.join(root_dir, '*.tif')):
        csv_path = img_path.replace('.tif', '.csv')
        if not os.path.exists(csv_path):
            csv_path = img_path.replace('.tif', '_consensus.csv')
        if not os.path.exists(csv_path):
            continue
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        df = pd.read_csv(csv_path)
        inst = []
        # expected columns
        cols = {c.lower(): c for c in df.columns}
        need = ['xmin', 'ymin', 'xmax', 'ymax']
        if not all(k in cols for k in need):
            # try alternative common casing
            continue
        label_col = None
        for cand in ['Cell Type', 'cell type', 'label', 'Label']:
            if cand in df.columns:
                label_col = cand
                break
        if label_col is None:
            continue
        for _, row in df.iterrows():
            try:
                x0 = int(row[cols['xmin']])
                y0 = int(row[cols['ymin']])
                x1 = int(row[cols['xmax']])
                y1 = int(row[cols['ymax']])
            except Exception:
                continue
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                continue
            label = str(row[label_col]).strip()
            inst.append({'bbox': (x0, y0, x1, y1), 'label': label})
        records.append({'image_path': img_path, 'width': width,
                       'height': height, 'instances': inst})
    return records


def read_midog_json(json_path: str, images_dir: str) -> List[Dict]:
    """Generic reader for MIDOG-style bbox json with image filenames and bboxes.
    Returns empty if files are missing.
    """
    records: List[Dict] = []
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        img_id_to_name = {d['id']: d['file_name']
                          for d in data.get('images', [])}
        img_id_to_wh = {d['id']: (d['width'], d['height'])
                        for d in data.get('images', [])}
        per_image: Dict[str, Dict] = {}
        for ann in data.get('annotations', []):
            img_id = ann.get('image_id')
            img_name = img_id_to_name.get(img_id)
            if not img_name:
                continue
            if img_name not in per_image:
                w, h = img_id_to_wh.get(img_id, (0, 0))
                per_image[img_name] = {
                    'image_path': os.path.join(images_dir, img_name),
                    'width': w, 'height': h,
                    'instances': []
                }
            # coco bbox: [x,y,w,h] -> convert to xyxy
            bb = ann.get('bbox', [0, 0, 0, 0])
            x0, y0, bw, bh = bb
            x1, y1 = x0 + bw, y0 + bh
            per_image[img_name]['instances'].append(
                {'bbox': (int(x0), int(y0), int(x1), int(y1)), 'label': 'mitoses'})
        records = list(per_image.values())
    except Exception:
        return []
    return records


def read_midog21(json_path: str, images_dir: str) -> List[Dict]:
    """Reader for MIDOG21 bbox json, fixed label 'mitoses'."""
    return read_midog_json(json_path, images_dir)


def read_breast_nucls(root_dir: str) -> List[Dict]:
    """Read breast_nucls nuclei bounding boxes from CSV/polyline/rect annotations.

    Folder layout under root_dir:
      train/rgb, train/csv, train/mask
      eval/rgb,  eval/csv,  eval/mask

    CSV fields expected:
      - type: 'rectangles' or 'polylines'
      - xmin, ymin, xmax, ymax (rectangles)
      - coords_x, coords_y (comma-separated for polylines)
      - main_classification or raw_classification: category name or id
    """
    def map_casemask_to_image(casemask: str) -> str:
        # from preprocess script: caseimg = casemask.split('#_')[-1]
        caseimg = casemask.split('#_')[-1]
        head = caseimg.split('_')[0]
        parts = head.split('-')
        if len(parts) == 6:
            tcgacase = '-'.join(parts[:3] + [parts[-1]])
            caseimg = '_'.join([tcgacase] + caseimg.split('_')[1:])
        return caseimg

    # labels to drop
    drop_names = {"fov", "unlabled", "unknown"}

    def collect_from_split(split: str) -> List[Dict]:
        rgb_dir = os.path.join(root_dir, split, 'rgb')
        csv_dir = os.path.join(root_dir, split, 'csv')
        records: Dict[str, Dict] = {}
        for csv_path in glob.glob(os.path.join(csv_dir, '*.csv')):
            casemask = os.path.basename(csv_path)[:-4]
            caseimg = map_casemask_to_image(casemask)
            img_path = os.path.join(rgb_dir, f'{caseimg}.png')
            if not os.path.exists(img_path):
                continue
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception:
                continue
            if img_path not in records:
                records[img_path] = {'image_path': img_path,
                                     'width': w, 'height': h, 'instances': []}
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            # normalize columns
            cols = {c.lower(): c for c in df.columns}
            for _, row in df.iterrows():
                t = str(row.get(cols.get('type'), 'rectangles')).lower()
                label = None
                if 'main_classification' in cols:
                    label = row.get(cols['main_classification'])
                elif 'raw_classification' in cols:
                    label = row.get(cols['raw_classification'])
                if label is None:
                    continue
                label = str(label).strip()
                if label.lower() in drop_names:
                    continue
                # bbox
                if t.startswith('rect') and all(k in cols for k in ['xmin', 'ymin', 'xmax', 'ymax']):
                    try:
                        x0 = int(row[cols['xmin']])
                        y0 = int(row[cols['ymin']])
                        x1 = int(row[cols['xmax']])
                        y1 = int(row[cols['ymax']])
                    except Exception:
                        continue
                else:
                    # polyline -> external bbox from coords_x/coords_y
                    if not all(k in cols for k in ['coords_x', 'coords_y']):
                        continue
                    try:
                        xs = [int(float(v)) for v in str(
                            row[cols['coords_x']]).split(',') if v != '']
                        ys = [int(float(v)) for v in str(
                            row[cols['coords_y']]).split(',') if v != '']
                        if not xs or not ys:
                            continue
                        x0, x1 = min(xs), max(xs)
                        y0, y1 = min(ys), max(ys)
                    except Exception:
                        continue
                # clamp
                x0 = max(0, min(x0, w - 1))
                x1 = max(0, min(x1, w - 1))
                y0 = max(0, min(y0, h - 1))
                y1 = max(0, min(y1, h - 1))
                if x1 <= x0 or y1 <= y0:
                    continue
                if (x1 - x0) < 4 or (y1 - y0) < 4:
                    continue
                records[img_path]['instances'].append(
                    {'bbox': (x0, y0, x1, y1), 'label': label})
        return [rec for rec in records.values() if rec['instances']]

    all_recs: List[Dict] = []
    for split in ['train', 'eval']:
        all_recs.extend(collect_from_split(split))
    return all_recs
