# https://zenodo.org/records/7500843



import os, glob, sys
import numpy as np
import pandas as pd
from PIL import Image
sys.path.append('./')
from utils.cfg import VLDATA_RAW


if __name__ == '__main__':
    _dir = f'{VLDATA_RAW}/tlymph'
    cell_types = set()
    for imagpath in glob.glob(f'{_dir}/*.tif'):
        labelpath = imagpath.replace('.tif', '.csv')
        if not os.path.exists(labelpath):
            labelpath = imagpath.replace('.tif', '_consensus.csv')
            assert os.path.exists(labelpath), labelpath
        img = Image.open(imagpath).convert('RGB')
        width, height = img.size
        label = pd.read_csv(labelpath)
        for celltype in label['Cell Type'].unique():
            cell_types.add(celltype)
        for it, row in label.iterrows():
            x0, y0, x1, y1 = int(row['XMin']), int(row['YMin']), int(row['XMax'])-1, int(row['YMax'])-1
            assert 0<=x0<width and 0<=x1<width, (x0, x1, width)
            assert 0<=y0<height and 0<=y1<height, (y0, y1, height)
        print(width, 'x', height, ':', len(label), 'cells' )
        
    print(cell_types) # {'Other cell', 'Diverse', 'CD3+ immune cell', 'Tumor cell'}