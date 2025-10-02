import numpy as np
import os
import json
import cv2
from PIL import Image
import multiprocessing as mp
from .segment_utils import segmentTissue
from tqdm import tqdm

class Contour_Checking_fn(object):
	# Defining __call__ method 
	def __call__(self, pt): 
		raise NotImplementedError

class isInContourV3_Easy(Contour_Checking_fn):
	def __init__(self, contour, patch_size, center_shift=0):
		self.cont = contour
		self.patch_size = patch_size
		self.shift = int(patch_size//2*center_shift)
	def __call__(self, pt): 
		center = (pt[0]+self.patch_size//2, pt[1]+self.patch_size//2)
		if self.shift > 0:
			all_points = [(center[0]-self.shift, center[1]-self.shift),
						  (center[0]+self.shift, center[1]+self.shift),
						  (center[0]+self.shift, center[1]-self.shift),
						  (center[0]-self.shift, center[1]+self.shift)
						  ]
		else:
			all_points = [center]
		
		for points in all_points:
			if cv2.pointPolygonTest(self.cont, tuple(np.array(points).astype(float)), False) >= 0:
				return 1
		return 0

def isInHoles(holes, pt, patch_size):
    for hole in holes:
        if cv2.pointPolygonTest(hole, (pt[0]+patch_size/2, pt[1]+patch_size/2), False) > 0:
            return 1
    return 0

def isInContours(cont_check_fn, pt, holes=None, patch_size=256):
    if cont_check_fn(pt):
        if holes is not None:
            return not isInHoles(holes, pt, patch_size)
        else:
            return 1
    return 0

def isPatchLabel(mask, ref_patch_size, label_thresh_dict):
    for label, thresh in label_thresh_dict.items():
        min_pixel = thresh*ref_patch_size*ref_patch_size
        if np.all(np.sum(mask==label, axis = (0,1))>=min_pixel):
            return label
    return None

def process_coord_candidate(whole_mask, label_thresh_dict, coord, contour_holes, ref_patch_size, cont_check_fn):
    # mask (H, W, n)
    if isInContours(cont_check_fn, coord, contour_holes, ref_patch_size):
        mask = whole_mask[coord[1]:coord[1]+ref_patch_size, coord[0]:coord[0]+ref_patch_size]
        label = isPatchLabel(mask, ref_patch_size, label_thresh_dict)
        if label is not None:
            return coord, label
    return None

def save_patch(whole_img, whole_mask, coord,tile_hw, outf, save_mask=False):
    y0, y1 = coord[1], coord[1]+tile_hw
    x0, x1 = coord[0], coord[0]+tile_hw
    img = whole_img[y0:y1,x0:x1]
    Image.fromarray(img).save(f'{outf}_{x0}_{y0}.png')
    # print(f'{outf}_{x0}_{y0}.png')
    if save_mask:
        mask = whole_mask[y0:y1,x0:x1]
        np.save(f'{outf}_{x0}_{y0}.npy', mask)

def crop_img_mask_wcountour(whole_img, whole_mask, out_dir, wsi_mark, label_def_dict, label_thresh_dict, tile_hw=224, save_mask=False):
    contours, holes = segmentTissue(whole_img)
    print(f'Found {len(contours)} contours and {len(holes)} holes')
    topleft_pts = []
    patch_labels = []
    num_workers = min(10, mp.cpu_count())
    print(num_workers)
    for nc, (contour, hole) in enumerate(zip(contours, holes)):
        # start_x, start_y, w, h = cv2.boundingRect(contour) 
        # stop_y = start_y+h
        # stop_x = start_x+w
        start_x, stop_x = contour[:,:,0].min(), contour[:,:,0].max()+1
        start_y, stop_y = contour[:,:,1].min(), contour[:,:,1].max()+1
        x_range = np.arange(start_x, stop_x, step=tile_hw)
        y_range = np.arange(start_y, stop_y, step=tile_hw)
        x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')
        coord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose() #(n, 2)
        pool = mp.Pool(num_workers)
        incontour_fn = isInContourV3_Easy(contour, tile_hw)
        iterable = [(whole_mask, label_thresh_dict, coord, hole, tile_hw, incontour_fn) for coord in coord_candidates]
        results = pool.starmap(process_coord_candidate, iterable)
        pool.close()
        topleft_pts += [result[0] for result in results if result is not None]
        patch_labels += [result[1] for result in results if result is not None]
        print(f'{len(patch_labels)}/{len(results)}({len(coord_candidates)}) patches in {nc}-th contour')

    # save patch
    print(f'Saving {len(patch_labels)} qualified patches')
    for label in label_def_dict.keys():
        os.makedirs(f'{out_dir}/{label}', exist_ok=True)
    with open(f"{out_dir}/label.json", "w") as json_file:
        json.dump(label_def_dict, json_file)

    save_mask = False
    pool = mp.Pool(num_workers)
    iterable = [(whole_img, whole_mask, coord, tile_hw, f'{out_dir}/{label}/{wsi_mark}', save_mask) for coord, label in zip(topleft_pts, patch_labels)]
    results = pool.starmap(save_patch, iterable)
    pool.close()  
    # breakpoint()

def crop_img_wcountour(whole_img, out_dir, wsi_mark, tile_hw=224, save_mask=False):
    contours, holes = segmentTissue(whole_img)
    print(f'Found {len(contours)} contours and {len(holes)} holes')
    topleft_pts = []
    num_workers = min(10, mp.cpu_count())
    print(num_workers)
    for nc, (contour, hole) in enumerate(zip(contours, holes)):
        # start_x, start_y, w, h = cv2.boundingRect(contour) 
        # stop_y = start_y+h
        # stop_x = start_x+w
        start_x, stop_x = contour[:,:,0].min(), contour[:,:,0].max()+1
        start_y, stop_y = contour[:,:,1].min(), contour[:,:,1].max()+1
        x_range = np.arange(start_x, stop_x, step=tile_hw)
        y_range = np.arange(start_y, stop_y, step=tile_hw)
        x_coords, y_coords = np.meshgrid(x_range, y_range, indexing='ij')
        coord_candidates = np.array([x_coords.flatten(), y_coords.flatten()]).transpose() #(n, 2)
        pool = mp.Pool(num_workers)
        incontour_fn = isInContourV3_Easy(contour, tile_hw)
        iterable = [(incontour_fn, coord, hole, tile_hw) for coord in coord_candidates]
        results = pool.starmap(isInContours, iterable)
        pool.close()
       
        topleft_pts += [coord for result, coord in zip(results,coord_candidates) if result]
        print(f'{len(topleft_pts)}/{len(results)}({len(coord_candidates)}) patches in {nc}-th contour')

    # save patch
    save_mask = False
    pool = mp.Pool(num_workers)
    iterable = [(whole_img, None, coord, tile_hw, f'{out_dir}/{wsi_mark}', save_mask) for coord in topleft_pts]
    results = pool.starmap(save_patch, iterable)
    pool.close()  

def label_patch(mask, label_thresh_dict, target_labels):
    height, width = mask.shape[:2]
    for label in target_labels:
        min_pixel = label_thresh_dict[label]*height*width
        if np.all(np.sum(mask==label, axis = (0,1))>=min_pixel):
            return label
    return None

def crop_img_mask(whole_img, whole_mask, out_dir, wsi_mark, label_def_dict, label_thresh_dict, tile_hw=224, save_mask=False):
    """
    whole_img: np.array (H, W, 3)
    whole_mask: np.array (H, W, n) n is the number of pathologiests
    out_dir: str
    wsi_mark: str
    label_def_dict: dict

    Output
    - save image in png
    - save mask in json
    """
    # output
    target_labels = label_def_dict.keys()
    for label in target_labels:
        os.makedirs(f'{out_dir}/{label}', exist_ok=True)
    with open(f"{out_dir}/label.json", "w") as json_file:
        json.dump(label_def_dict, json_file)
    #
    height, width, _ = whole_img.shape
    assert whole_mask.shape[:2] == (height, width)
    num_x = width//tile_hw
    num_y = height//tile_hw
    qualified_cnt = 0
    for y in range(num_y):
        for x in range(num_x):
            y0, y1 = y*tile_hw, (y+1)*tile_hw
            x0, x1 = x*tile_hw, (x+1)*tile_hw

            img = whole_img[y0:y1,x0:x1]
            mask = whole_mask[y0:y1,x0:x1]
            label = label_patch(mask,
                                label_thresh_dict=label_thresh_dict,
                                target_labels=target_labels)
            if label is not None:
                Image.fromarray(img).save( f'{out_dir}/{label}/{wsi_mark}_{x0}_{y0}.png')
                qualified_cnt +=1
                if save_mask:
                    np.save(f'{out_dir}/{label}/{wsi_mark}_{x0}_{y0}.npy', mask)
    print(f'{qualified_cnt}/{num_x*num_y} qualified patches')


def _save_img_caption(whole_img, whole_mask, bbox, out_dir, wsi_mark, label_def_dict, label_thresh_dict, label_centroids):
    x0, y0, x1, y1 = bbox
    img = whole_img[y0:y1,x0:x1]
    mask = whole_mask[y0:y1,x0:x1]
    
    caption = {}
    for gt, cap in label_def_dict.items():
        percent = np.sum(mask==gt)/((x1-x0)*(y1-y0))
        caption[cap] = round(percent,4)
        if label_thresh_dict:
            if percent < label_thresh_dict[gt]:
                return {}
    
    imgpath = f'{out_dir}/{wsi_mark}_{x0}_{y0}.png'
    Image.fromarray(img).save(imgpath)
    if label_centroids:
        count = {}
        for gt, cap in label_def_dict.items():
            if label_centroids[gt]:
                cnt = 0
                for xy in label_centroids[gt]:
                    if x0<=xy[0]<=x1 and y0<=xy[1]<=y1:
                        cnt +=1
                count[cap] = cnt
        return {'image':imgpath, 'label':caption, 'count':count}
    else:
        return {'image':imgpath, 'label':caption}

def crop_img_mask_to_caption(whole_img, whole_mask, out_dir, wsi_mark, label_def_dict, tile_hw=224, label_thresh_dict={}, label_centroids={}):
    """
    whole_img: np.array (H, W, 3)
    whole_mask: np.array (H, W, n) n is the number of pathologiests
    out_dir: str
    wsi_mark: str
    label_def_dict: dict

    Output
    - save image in png
    - save caption in json
    """
    target_labels = label_def_dict.keys()
    # for label in target_labels:
    #     os.makedirs(f'{out_dir}/{label}', exist_ok=True)
    # with open(f"{out_dir}/label.json", "w") as json_file:
    #     json.dump(label_def_dict, json_file)
    #
    height, width, _ = whole_img.shape
    assert whole_mask.shape[:2] == (height, width)
    num_x = width//tile_hw
    num_y = height//tile_hw
    coords = []
    for y in range(num_y):
        for x in range(num_x):
            y0, y1 = y*tile_hw, (y+1)*tile_hw
            x0, x1 = x*tile_hw, (x+1)*tile_hw
            coords.append((x0, y0, x1, y1))
    # save cropped in parallel
    num_workers = min(10, mp.cpu_count())
    pool = mp.Pool(num_workers)
     
    iterable = [(whole_img, whole_mask, coord, out_dir, wsi_mark, label_def_dict, label_thresh_dict, label_centroids) for coord in coords]
    results = pool.starmap(_save_img_caption, iterable)
    pool.close()  
    
    print(f'{wsi_mark}')
    return results