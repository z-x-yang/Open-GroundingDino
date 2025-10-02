import os
import glob
from typing import List, Dict, Tuple

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
                x0 = int(row[cols['xmin']]); y0 = int(row[cols['ymin']])
                x1 = int(row[cols['xmax']]); y1 = int(row[cols['ymax']])
            except Exception:
                continue
            if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
                continue
            label = str(row[label_col]).strip()
            inst.append({'bbox': (x0, y0, x1, y1), 'label': label})
        records.append({'image_path': img_path, 'width': width, 'height': height, 'instances': inst})
    return records


