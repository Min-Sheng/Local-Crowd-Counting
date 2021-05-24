import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.spatial
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from pycocotools.coco import COCO

sys.path.append('../')
from get_density_map_gaussian import get_density_map_gaussian

def cocoBboxToDensityMap(coco, imgId, includeCrowd=False, excludeInvalid=False):
	'''
	Convert COCO GT or results for a single image to a density map.
	:param coco: an instance of the COCO API (ground-truth or result)
	:param imgId: the id of the COCO image
	:param includeCrowd: whether to include 'crowd' thing annotations as 'other' (or void)
	:param excludeInvalid: whether to exclude 'invalid' class
	:return: labelMap - [h x w] density map
	'''
	# Init
	curImg = coco.imgs[imgId]
	imageSize = (curImg['height'], curImg['width'])

	# Get annotations of the current image (may be empty)
	if includeCrowd:
		annIds = coco.getAnnIds(imgIds=imgId)
	else:
		annIds = coco.getAnnIds(imgIds=imgId, iscrowd=False)
	imgAnnots = coco.loadAnns(annIds)

	gt = [imgAnnot['bbox'] + [imgAnnot['category_id']] for imgAnnot in imgAnnots]

	center = []
	for i in range(0,len(gt)):
		x, y, w, h, cat = gt[i]
		if cat == coco.getCatIds(['Invalid'])[0] and excludeInvalid:
			continue
		else:
			x_center = x + w //2
			y_center = y + h //2
			center.append([x_center, y_center])
	center = np.array(center)

	# print('gt sum:', len(center))

	# generation
	im_density = get_density_map_gaussian(imageSize, center, 11, 2)
	# print('den sum: ', im_density.sum(axis=(0, 1)))

	return im_density


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--coco_json_path', required=False, default='/mnt/dataset/NTUH_HEMA/63x/hema_cocoformat_train.json', type=str)
	parser.add_argument('--exclude_invalid', action='store_true')
	parser.add_argument('--k_size', required=False, default=15, type=int)
	parser.add_argument('--sigma', required=False, default=4, type=int)
	args = parser.parse_args() 
	# Initialize COCO ground-truth API
	coco = COCO(args.coco_json_path)
	
	imgIds = coco.getImgIds()
	imgIds = sorted(set([a['image_id'] for a in coco.anns.values()]))
	
	imgCount = len(imgIds)
	
	if "33class" in args.coco_json_path:
		print("Parsing in 33 classes")
		class33 = True
	else:
		class33 = False
		
	for i in tqdm(range(imgCount)):
		imgId = imgIds[i]
		if class33:
			if args.exclude_invalid:
				densityMapPath = coco.loadImgs(ids=imgId)[0]['file_name'].replace('.tiff', f'_density_class33_k{args.k_size}_sigma{args.sigma}_exclude_invalid.npy')
			else:
				densityMapPath = coco.loadImgs(ids=imgId)[0]['file_name'].replace('.tiff', f'_density_class33_k{args.k_size}_sigma{args.sigma}.npy')
		else:
			if args.exclude_invalid:
				densityMapPath = coco.loadImgs(ids=imgId)[0]['file_name'].replace('.tiff', f'_density_k{args.k_size}_sigma{args.sigma}_exclude_invalid.npy')
			else:
				densityMapPath = coco.loadImgs(ids=imgId)[0]['file_name'].replace('.tiff', f'_density_k{args.k_size}_sigma{args.sigma}.npy')

		#if not os.path.exists(segmentationPath):
		d = cocoBboxToDensityMap(coco, imgId, args.exclude_invalid)
		d = d.astype(np.float32)
		np.save(densityMapPath, d)

