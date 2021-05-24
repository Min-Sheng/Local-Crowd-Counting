import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO

# TODO 

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_json_path', required=False, default='/mnt/dataset/NTUH_HEMA/31p5x/hema_cocoformat_train.json', type=str)

    return parser

if __name__ == '__main__':
    args = make_parser().parse_args()
    coco = COCO(args.coco_json_path)

    imgs_list = []

    img_ids = coco.getImgIds()
    img_ids = sorted(set([a['image_id'] for a in coco.anns.values()]))

    img_count = len(img_ids)
    
    for i in tqdm(range(img_count)):
        img_id = img_ids[i]
                    
        img_path = coco.loadImgs(ids=img_id)[0]['file_name']

        img = Image.open(img_path)
        if img.mode == 'L':
            img = img.convert('RGB')

        img = np.array(img.resize((172, 115),Image.BILINEAR))

        imgs_list.append(img)

    imgs = np.array(imgs_list).astype(np.float32)/255.
    red = imgs[:,:,:,0]
    green = imgs[:,:,:,1]
    blue = imgs[:,:,:,2]


    print("means: [{}, {}, {}]".format(np.mean(red),np.mean(green),np.mean(blue)))
    print("stdevs: [{}, {}, {}]".format(np.std(red),np.std(green),np.std(blue)))
