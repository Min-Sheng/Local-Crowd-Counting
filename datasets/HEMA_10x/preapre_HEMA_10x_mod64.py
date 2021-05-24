import os
import sys
import cv2
import json
import numpy as np

sys.path.append('../')
from get_density_map_gaussian import get_density_map_gaussian

dataset = ['train']

train_data_json = '../../ProcessedData/hema_10x/train.json'
val_data_json = '../../ProcessedData/hema_10x/val.json'
test_data_json= '../../ProcessedData/hema_10x/test.json'

ann_dir = '/mnt/deep-learning/usr/seanyu/home/storage/hema-lowres-detection/csv'

output_path = '../../ProcessedData/hema_10x/'
if not os.path.exists(output_path):
	os.mkdir(output_path)

if 'test' in dataset:

	# test set
	output_path += 'test/'
	path_img = output_path + 'img/'
	path_den = output_path + 'den/'

	if not os.path.exists(output_path):
		os.mkdir(output_path)
	if not os.path.exists(path_img):
		os.mkdir(path_img)
	if not os.path.exists(path_den):
		os.mkdir(path_den)
	
	with open(test_data_json) as f:
		data = json.load(f)

	for i, slide in enumerate(data):
		
		file_name = os.path.basename(slide).split('.')[0]

		print("idx: ", i)

		if not os.path.exists(f'{ann_dir}/{file_name}.csv'):
			continue
			
		if os.path.exists(f'{path_den}{file_name}.npy'):
			continue

		# load gt
		center = np.genfromtxt(f'{ann_dir}/{file_name}.csv', delimiter=',', dtype=np.int)
		print('gt sum:', len(center))

		# load img
		img = cv2.imread(slide)

		# resize
		img_1p25x = cv2.resize(img, (int(img.shape[1]*0.125), int(img.shape[0]*0.125)))
		center_1p25x = center * 0.125

		# generation
		im_density = get_density_map_gaussian(img_1p25x.shape[:-1], center_1p25x, 9, 1)
		print('den sum: ', im_density.sum(axis=(0, 1)))

		# save img
		cv2.imwrite(f'{path_img}{file_name}.tiff', img_1p25x)

		# save npy
		im_density = im_density.astype(np.float32)
		np.save(f'{path_den}{file_name}.npy', im_density)

if 'val' in dataset:

	# test set
	output_path += 'val/'
	path_img = output_path + 'img/'
	path_den = output_path + 'den/'

	if not os.path.exists(output_path):
		os.mkdir(output_path)
	if not os.path.exists(path_img):
		os.mkdir(path_img)
	if not os.path.exists(path_den):
		os.mkdir(path_den)
	
	with open(val_data_json) as f:
		data = json.load(f)

	for i, slide in enumerate(data):
		
		print("idx: ", i)

		file_name = os.path.basename(slide).split('.')[0]

		if not os.path.exists(f'{ann_dir}/{file_name}.csv'):
			continue
		
		if os.path.exists(f'{path_den}{file_name}.npy'):
			continue

		# load gt
		center = np.genfromtxt(f'{ann_dir}/{file_name}.csv', delimiter=',', dtype=np.int)
		print('gt sum:', len(center))

		# load img
		img = cv2.imread(slide)

		# resize
		img_1p25x = cv2.resize(img, (int(img.shape[1]*0.125), int(img.shape[0]*0.125)))
		center_1p25x = center * 0.125

		# generation
		im_density = get_density_map_gaussian(img_1p25x.shape[:-1], center_1p25x, 9, 1)
		print('den sum: ', im_density.sum(axis=(0, 1)))

		# save img
		cv2.imwrite(f'{path_img}{file_name}.tiff', img_1p25x)

		# save npy
		im_density = im_density.astype(np.float32)
		np.save(f'{path_den}{file_name}.npy', im_density)

if 'train' in dataset:

	# test set
	output_path += 'train/'
	path_img = output_path + 'img/'
	path_den = output_path + 'den/'

	if not os.path.exists(output_path):
		os.mkdir(output_path)
	if not os.path.exists(path_img):
		os.mkdir(path_img)
	if not os.path.exists(path_den):
		os.mkdir(path_den)
	
	with open(train_data_json) as f:
		data = json.load(f)

	for i, slide in enumerate(data):
		
		print("idx: ", i)

		file_name = os.path.basename(slide).split('.')[0]

		if not os.path.exists(f'{ann_dir}/{file_name}.csv'):
			continue

		if os.path.exists(f'{path_den}{file_name}.npy'):
			continue

		# load gt
		center = np.genfromtxt(f'{ann_dir}/{file_name}.csv', delimiter=',', dtype=np.int)
		print('gt sum:', len(center))

		# load img
		img = cv2.imread(slide)

		# resize
		img_1p25x = cv2.resize(img, (int(img.shape[1]*0.125), int(img.shape[0]*0.125)))
		center_1p25x = center * 0.125

		# generation
		im_density = get_density_map_gaussian(img_1p25x.shape[:-1], center_1p25x, 9, 1)
		print('den sum: ', im_density.sum(axis=(0, 1)))

		# save img
		cv2.imwrite(f'{path_img}{file_name}.tiff', img_1p25x)

		# save npy
		im_density = im_density.astype(np.float32)
		np.save(f'{path_den}{file_name}.npy', im_density)