# https://monusac-2020.grand-challenge.org/Data/
import numpy as np
import glob
import os
import openslide
from PIL import Image
# import xmltodict
# from shapely.geometry import Polygon
import xml.etree.ElementTree as ET
from skimage import draw
import cv2
from tqdm import tqdm
import json
import sys
sys.path.append('./')
from utils.cfg import VLDATA_RAW, VLDATA_PROCESS
from preprocess.tile_utils import crop_img_mask_to_caption
from utils.draw import counts_and_draw


Image.MAX_IMAGE_PIXELS = None

_VIS_DIR = f'{VLDATA_PROCESS}/visualization/monusac'
os.makedirs(_VIS_DIR, exist_ok=True)

_RAW_DIR = f'{VLDATA_RAW}/monusac'
_PROC_DIR = f'{VLDATA_RAW}/monusac_processed'
os.makedirs(_PROC_DIR, exist_ok=True)


def xml_to_imag_mask_png(wsi, xmlf, sub_image, gt=255):
    image = wsi.read_region((0,0),0, wsi.level_dimensions[0])
    image.convert('RGB').save(sub_image+'/image.png')
    try:
        image = Image.open(sub_image+'/image.png')
    except:
        breakpoint()
    count = 0
    # Read xml file
    tree = ET.parse(xmlf)
    root = tree.getroot()
    #Generate n-ary mask for each cell-type                         
    for k in range(len(root)):
        label = [x.attrib['Name'] for x in root[k][0]]
        label = label[0]
        
        for child in root[k]:
            for x in child:
                r = x.tag
                if r == 'Attribute':
                    count = count+1
                    label = x.attrib['Name']

                    n_ary_mask = np.transpose(np.zeros((image.size))) 
                    
                    # Create directory for each label
                    sub_path = sub_image+'/'+label 
                    
                    # try:
                    #     os.makedirs(sub_path, exist_ok=True)
                    # except OSError:
                    #     print ("Creation of the directory %s failed" % label)
                    # else:
                    #     print ("Successfully created the directory %s " % label) 
                                        
                    
                if r == 'Region':
                    regions = []
                    vertices = x[1]
                    coords = np.zeros((len(vertices), 2))
                    for i, vertex in enumerate(vertices):
                        coords[i][0] = vertex.attrib['X']
                        coords[i][1] = vertex.attrib['Y']        
                    regions.append(coords)
                    # poly = Polygon(regions[0])  
                    
                    vertex_row_coords = regions[0][:,0]
                    vertex_col_coords = regions[0][:,1]
                    fill_row_coords, fill_col_coords = draw.polygon(vertex_col_coords, vertex_row_coords, n_ary_mask.shape)
                    # gt = gt+1 #Keep track of giving unique valu to each instance in an image
                    n_ary_mask[fill_row_coords, fill_col_coords] = gt # binary
                    mask_path = sub_path+ '_' +str(count)+'_mask.png'
                    cv2.imwrite(mask_path, n_ary_mask)
                    # print(mask_path, label, n_ary_mask.shape)

    return count, image.size

def extract_mask(basedir, outdir):
    image_sizes = []
    imgfs = glob.glob(f'{basedir}/*/*.svs')
    for imgf in tqdm(imgfs):
        # img = Image.open(imgf)
        wsi = openslide.OpenSlide(imgf)
        xmlf = imgf[:-4] + '.xml'
        patient_id = imgf.split('/')[-2]
        sub_image_name = imgf.split('/')[-1][:-4]
        sub_image = f'{outdir}/{sub_image_name}'
        
        os.makedirs(sub_image, exist_ok=True)
        cnt, image_size = xml_to_imag_mask_png(wsi, xmlf, sub_image)
        image_sizes.append(image_size)
        print(cnt)
        # tiff = imgf[:-4] + '.tif'
        # print(Image.open(tiff).size)
        # shutil.copy(tiff, f'{sub_image}/image.tif')
        # breakpoint()
        # # If svs image needs to save in tif
        # cv2.imwrite(, wsi.array(img.read_region((0,0),0,img.level_dimensions[0])))              
    print(set(image_sizes))

def collect_imgmask(basedir):
    imgfs = sorted(glob.glob(f'{basedir}/*/*.tif'))
    maskfs = sorted(glob.glob(f'{basedir}/*/*_mask.png'))
    for imgf, maskf in zip(imgfs, maskfs):
        assert os.path.dirname(imgf) == os.path.dirname(maskf)
        img = np.array(Image.open(imgf).convert("RGB"))
        mask = np.array(Image.open(maskf))
        print(img.shape, mask.shape, np.unique(mask))
        breakpoint()


if __name__ == '__main__':
    # Image and mask with size varing from 100 to 5000

    # 1. Preprocess mask: XML to mask png
    for subdir in os.listdir(_RAW_DIR):
        if subdir=="MoNuSAC_Testing_Color_Coded_Masks":
            continue
        split = 'Testing' if 'Testing' in subdir else 'Training'
        outdir =  f'{_PROC_DIR}/{split}_image_mask'
        if not os.path.exists(outdir):
            os.makedirs(outdir, exist_ok=True)
            extract_mask(f'{_RAW_DIR}/{subdir}',outdir)
    
    # 2. Crop: Collect image and masks -> crop to 336x336
    jsonf = f'{VLDATA_PROCESS}/mix_monusac.json'
    crop_dir = f'{VLDATA_PROCESS}/monusac_cropped'
    os.makedirs(crop_dir, exist_ok=True)

    imgfs = glob.glob(f'{_PROC_DIR}/Training_image_mask/*/image.png') + glob.glob(f'{_PROC_DIR}/Testing_image_mask/*/image.png')
    labels = {'Ambiguous':1,
            'Epithelial': 2,
            'Macrophage':3,
            'Neutrophil':4,
            'Lymphocyte':5,
            }
    if not os.path.exists(jsonf):
        datalist = []
        for imgf in tqdm(imgfs):
            img = np.array(Image.open(imgf))
            # 
            maskfs = glob.glob(f'{os.path.dirname(imgf)}/*mask.png')
            gts = [labels[os.path.basename(maskf).split('_')[0]] for maskf in maskfs]
            # rank the labels via its importance, there will be overwrite combining into one mask
            gts, maskfs = zip(*sorted(zip(gts, maskfs)))
            mask = np.zeros(img.shape[:2])
            for maskf in maskfs:
                label = os.path.basename(maskf).split('_')[0]
                cur_mask = np.array(Image.open(maskf)) #mask: 0 or 255
                fg_index = cur_mask>0
                if not np.all(mask[fg_index]==0):
                    print(np.unique(mask[fg_index]), label, 'overlap')
                mask[fg_index] = labels[label]
            # 
            standard_hw = 336
            height, width = img.shape[:2]
            num_width, num_height = width//standard_hw+1, height//standard_hw+1
            # num_width, num_height = max(num_width, 1), max(1, num_height)
            tile_hw = min(height//num_height, width//num_width)
            num_tile = (height//tile_hw)*(width//tile_hw)
            print(height, width, tile_hw, '#',num_tile)
            results = crop_img_mask_to_caption(whole_img=img,
                        whole_mask=mask[:,:,None],
                        out_dir=crop_dir,
                        wsi_mark=imgf.split('/')[-2],
                        label_def_dict={label: deff for deff,label in labels.items()}, 
                        tile_hw=tile_hw,
                        save_mask=True)
            # Transform the mask to instance count
            for r in results:
                mask = np.load(r['mask'])
                
                counts = counts_and_draw(image = np.array(Image.open(r['image'])),
                                        multiclass_mask=mask,
                                        label_defs = {label: deff for deff,label in labels.items()},
                                        vis_imgf = f'{_VIS_DIR}/{os.path.basename(r["image"])}')
                r.update({'count': {k: float(v) for k, v in counts.items()}})
                r.pop('mask')
            datalist += results
        print(len(datalist))
        with open(jsonf, 'w') as json_file:
            json.dump(datalist, json_file, indent=4)
    with open(jsonf) as f:
        data = json.load(f)
    breakpoint()