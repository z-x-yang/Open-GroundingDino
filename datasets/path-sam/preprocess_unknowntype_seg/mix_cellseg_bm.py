# https://zenodo.org/records/10719375


import os, glob, sys
import numpy as np
import pandas as pd
from PIL import Image
sys.path.append('./')
from utils.cfg import VLDATA_RAW

# Binary cell segmentation
# VLDATA_RAW = '/Users/gongxuan/Downloads'
def read_train_pairs(subfolder='Training-labeled'):
    img_dir = f'{_dir}/{subfolder}/images'
    label_dir = f'{_dir}/{subfolder}/labels'
    image_names= [f'cell_{n:05d}.bmp' for n in range(1,13)] + [f'cell_{n:05d}.bmp' for n in range(15,142)]
    image_names += [f'cell_{n:05d}.png' for n in range(145,229)]
    for image_name in image_names:
        img_path = f'{img_dir}/{image_name}'
        label_path = img_path.replace(img_dir, label_dir).replace('.png', '_label.tiff').replace('.bmp', '_label.tiff')
        image = np.array(Image.open(img_path))
        mask = np.array(Image.open(label_path))
        height, width = mask.shape[:2]
        assert image.shape == (height, width, 3), (image.shape, (height, width, 3))
        print(height, width, len(np.unique(mask)), 'instances')

def read_tune_pairs(subfolder='Tuning'):
    img_dir = f'{_dir}/{subfolder}/images'
    label_dir = f'{_dir}/{subfolder}/labels'
    image_names= [f'cell_{n:05d}' for n in range(1,27)] 
    for image_name in image_names:
        img_path = glob.glob(f'{img_dir}/{image_name}*')
        assert len(img_path)==1, img_path
        img_path = img_path[0]
        label_path = img_path.replace(img_dir, label_dir).replace('.tiff', '_label.tiff').replace('.png', '_label.tiff')
        image = np.array(Image.open(img_path))
        mask = np.array(Image.open(label_path))
        height, width = mask.shape[:2]
        assert image.shape == (height, width, 3), (image.shape, (height, width, 3))
        print(height, width, len(np.unique(mask)), 'instances')

def read_test_pairs_1(subfolder='Testing/Hidden'):
    img_dir = f'{_dir}/{subfolder}/filter_images'
    label_dir = f'{_dir}/{subfolder}/osilab_seg' #Prediction from osiliab, not label!
    image_names= ['TestHidden_356.tif', 'TestHidden_381.tif', 'TestHidden_168.tif', 'TestHidden_008.tif', 'TestHidden_377.bmp', 'TestHidden_020.tif', 'TestHidden_222.tif', 'TestHidden_236.tif', 'TestHidden_389.bmp', 'TestHidden_035.tif', 'TestHidden_362.bmp', 'TestHidden_021.tif', 'TestHidden_169.tif', 'TestHidden_394.tif', 'TestHidden_028.bmp', 'TestHidden_341.tif', 'TestHidden_016.bmp', 'TestHidden_348.bmp', 'TestHidden_037.tif', 'TestHidden_189.bmp', 'TestHidden_177.bmp', 'TestHidden_022.tif', 'TestHidden_349.bmp', 'TestHidden_181.tif', 'TestHidden_185.tif', 'TestHidden_224.tif', 'TestHidden_033.tif', 'TestHidden_147.tif', 'TestHidden_386.tif', 'TestHidden_012.bmp', 'TestHidden_353.tif', 'TestHidden_192.tif', 'TestHidden_031.tif', 'TestHidden_232.tif', 'TestHidden_226.tif', 'TestHidden_018.tif', 'TestHidden_193.tif', 'TestHidden_005.bmp', 'TestHidden_076.bmp', 'TestHidden_260.bmp', 'TestHidden_043.tif', 'TestHidden_094.tif', 'TestHidden_116.bmp', 'TestHidden_241.tif', 'TestHidden_102.bmp', 'TestHidden_254.tif', 'TestHidden_336.tif', 'TestHidden_120.tif', 'TestHidden_134.tif', 'TestHidden_317.bmp', 'TestHidden_242.tif', 'TestHidden_281.tif', 'TestHidden_294.tif', 'TestHidden_257.tif', 'TestHidden_272.bmp', 'TestHidden_119.tif', 'TestHidden_044.tif', 'TestHidden_050.tif', 'TestHidden_059.bmp', 'TestHidden_326.tif', 'TestHidden_271.bmp', 'TestHidden_339.bmp', 'TestHidden_085.tif', 'TestHidden_279.tif', 'TestHidden_047.tif', 'TestHidden_133.tif', 'TestHidden_328.tif', 'TestHidden_062.tif', 'TestHidden_335.bmp', 'TestHidden_248.tif', 'TestHidden_123.bmp', 'TestHidden_063.tif', 'TestHidden_283.bmp', 'TestHidden_329.tif', 'TestHidden_303.tif', 'TestHidden_115.tif', 'TestHidden_316.tif', 'TestHidden_306.tif', 'TestHidden_079.bmp', 'TestHidden_086.bmp', 'TestHidden_070.tif', 'TestHidden_332.bmp', 'TestHidden_078.bmp', 'TestHidden_244.bmp', 'TestHidden_098.tif', 'TestHidden_127.bmp', 'TestHidden_099.tif', 'TestHidden_331.bmp', 'TestHidden_292.bmp', 'TestHidden_286.bmp', 'TestHidden_112.tif', 'TestHidden_084.bmp', 'TestHidden_388.tif', 'TestHidden_149.tif', 'TestHidden_161.tif', 'TestHidden_342.bmp', 'TestHidden_216.tif', 'TestHidden_380.bmp', 'TestHidden_357.bmp', 'TestHidden_023.bmp', 'TestHidden_235.bmp', 'TestHidden_162.tif', 'TestHidden_002.tif', 'TestHidden_396.bmp', 'TestHidden_228.tif', 'TestHidden_214.tif', 'TestHidden_157.bmp', 'TestHidden_201.tif', 'TestHidden_215.tif', 'TestHidden_188.tif', 'TestHidden_163.tif', 'TestHidden_361.tif', 'TestHidden_365.tif', 'TestHidden_007.tif', 'TestHidden_239.tif', 'TestHidden_238.tif', 'TestHidden_170.tif', 'TestHidden_164.tif', 'TestHidden_347.bmp', 'TestHidden_186.bmp', 'TestHidden_398.tif', 'TestHidden_373.tif'] 
    for image_name in image_names:
        img_path = glob.glob(f'{img_dir}/{image_name}*')
        assert len(img_path)==1, img_path
        img_path = img_path[0]
        label_path = img_path.replace(img_dir, label_dir).replace('.tif', '_label.tiff').replace('.bmp', '_label.tiff')
        image = np.array(Image.open(img_path))
        mask = np.array(Image.open(label_path))
        height, width = mask.shape[:2]
        assert image.shape == (height, width, 3), (image.shape, (height, width, 3))
        print(height, width, len(np.unique(mask)), 'instances')


def read_test_pairs(subfolder='Testing/Public'):
    img_dir = f'{_dir}/{subfolder}/images'
    label_dir = f'{_dir}/{subfolder}/labels'
    image_names= [f'OpenTest_{n:03d}' for n in [3,5,10,12,18,20,22,24,27,29,30,34,36,37,38,40,49]] 
    for image_name in image_names:
        img_path = glob.glob(f'{img_dir}/{image_name}*')
        assert len(img_path)==1, img_path
        img_path = img_path[0]
        label_path = img_path.replace(img_dir, label_dir).replace('.tif', '_label.tiff').replace('.bmp', '_label.tiff')
        image = np.array(Image.open(img_path))
        mask = np.array(Image.open(label_path))
        height, width = mask.shape[:2]
        assert image.shape == (height, width, 3), (image.shape, (height, width, 3))
        print(height, width, len(np.unique(mask)), 'instances')

if __name__ == '__main__':
   _dir = f'{VLDATA_RAW}/cellseg'
   read_train_pairs()
   read_tune_pairs()
   read_test_pairs()