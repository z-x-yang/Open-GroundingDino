

import sys
sys.path.append('./')
from utils.cfg import VLDATA_RAW
from mix_midog22_b import read_image_bbox


if __name__ == '__main__':
    read_image_bbox(jsonf=f'{VLDATA_RAW}/MIDOG21/MIDOG.json',
               image_dir=f'{VLDATA_RAW}/MIDOG21/images',
               vis_dir = f'{VLDATA_RAW}/MIDOG21/visualize',)