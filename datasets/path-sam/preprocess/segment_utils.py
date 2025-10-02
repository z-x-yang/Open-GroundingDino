
import numpy as np
from PIL import Image
import cv2
import glob
from tqdm import tqdm

def _filter_contours(contours, hierarchy, filter_params):
        """
            Filter contours by: area.
        """
        filtered = []

        # find indices of foreground contours (parent == -1)
        hierarchy_1 = np.flatnonzero(hierarchy[:,1] == -1)
        all_holes = []
        
        # loop through foreground contour indices
        for cont_idx in hierarchy_1:
            # actual contour
            cont = contours[cont_idx]
            # indices of holes contained in this contour (children of parent contour)
            holes = np.flatnonzero(hierarchy[:, 1] == cont_idx)
            # take contour area (includes holes)
            a = cv2.contourArea(cont)
            # calculate the contour area of each hole
            hole_areas = [cv2.contourArea(contours[hole_idx]) for hole_idx in holes]
            # actual area of foreground contour region
            a = a - np.array(hole_areas).sum()
            if a == 0: continue
            if tuple((filter_params['a_t'],)) < tuple((a,)): 
                filtered.append(cont_idx)
                all_holes.append(holes)


        foreground_contours = [contours[cont_idx] for cont_idx in filtered]
        
        hole_contours = []

        for hole_ids in all_holes:
            unfiltered_holes = [contours[idx] for idx in hole_ids ]
            unfilered_holes = sorted(unfiltered_holes, key=cv2.contourArea, reverse=True)
            # take max_n_holes largest holes by area
            unfilered_holes = unfilered_holes[:filter_params['max_n_holes']]
            filtered_holes = []
            
            # filter these holes
            for hole in unfilered_holes:
                if cv2.contourArea(hole) > filter_params['a_h']:
                    filtered_holes.append(hole)

            hole_contours.append(filtered_holes)

        return foreground_contours, hole_contours

#  {'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False, 'keep_ids': 'none', 'exclude_ids': 'none'}
def segmentTissue(img, sthresh=20, sthresh_up = 255, mthresh=7, close = 4, use_otsu=False, 
                    filter_params={'a_t': 100.0, 'a_h': 16.0, 'max_n_holes': 8}, ref_patch_size=336):
        """
            Segment the tissue via HSV -> Median thresholding -> Binary threshold
        """
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space
        img_med = cv2.medianBlur(img_hsv[:,:,1], mthresh)  # Apply median blurring
        
        # Thresholding
        if use_otsu:
            _, img_otsu = cv2.threshold(img_med, 0, sthresh_up, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        else:
            _, img_otsu = cv2.threshold(img_med, sthresh, sthresh_up, cv2.THRESH_BINARY)

        # Morphological closing
        if close > 0:
            kernel = np.ones((close, close), np.uint8)
            img_otsu = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)                 


        filter_params = filter_params.copy()
        filter_params['a_t'] = filter_params['a_t'] * ref_patch_size
        filter_params['a_h'] = filter_params['a_h'] * ref_patch_size
        
        # Find and filter contours
        contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # Find contours 
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
        
        contours_tissue, holes_tissue = _filter_contours(contours, hierarchy, filter_params)  # Necessary for filtering out artifacts
        return contours_tissue, holes_tissue

@staticmethod
def scaleContourDim(contours, scale):
    return [np.array(cont * scale, dtype='int32') for cont in contours]

@staticmethod
def scaleHolesDim(contours, scale):
    return [[np.array(hole * scale, dtype = 'int32') for hole in holes] for holes in contours]

def draw_seg(img, contours, holes):
        holes = scaleHolesDim(holes, 1)
        contours = scaleContourDim(contours, 1)

        top_left = (0,0)
        offset = tuple(-(np.array(top_left)).astype(int))
        hole_color = (255,0,0)
        color = (0, 255, 0)
        line_thickness = 8
        for hole in holes:
            cv2.drawContours(img, hole, -1, hole_color, line_thickness, lineType=cv2.LINE_8)
        cv2.drawContours(img, contours, -1, color, line_thickness, lineType=cv2.LINE_8, offset=offset)
        # breakpoint()



if __name__ == "__main__":
    imgfs = glob.glob(f'/n/lw_groups/hms/dbmi/yu/lab/seg_data_raw/digestpath19_processed/Colonoscopy_tissue_segment_dataset/tissue-train-neg/*.jpg')
    outdir = '/n/lw_groups/hms/dbmi/yu/lab/xug751/code/pathx/debug'
    
    for imgf in tqdm(imgfs):
        img = cv2.imread(imgf)
        imgname = imgf.split('/')[-1]
        contours, holes = segmentTissue(img)
        draw_seg(img, contours, holes)
        Image.fromarray(img).save(f'{outdir}/{imgname}')