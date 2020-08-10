# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:05:56 2020

@author: journe
"""


from glob import glob
import os.path as op
import numpy as np
from skimage import io
import cv2
import sys
import os
from utils import resize_label, filter_nuclei, filter_site, load_boxes, Image_Reader
import time
from scipy.ndimage import distance_transform_edt
import pickle
import pandas as pd
import json
from tqdm import tqdm
EXPERIMENT_PATH ="E:\\Adrien\\data\\Experiment"


if __name__ == '__main__':
    print('Starting Post Analysis')
    start = time.time()
    data_path = sys.argv[1]
    annotation_path = sys.argv[2]
    
    _,exp_name = op.split(data_path)
    ir = Image_Reader(data_path)
    # train on the GPU or on the CPU, if a GPU is not available
    with open(op.join(annotation_path,'info_channel.json'), "w") as f:
        json.dump(ir.meta_channel,f)
    results = []
    for pos in tqdm(list(ir.pos.keys())):
        img = []
        for channel in ir.meta_channel.keys():
            img_channel = ir.get_image(pos,channel)
            img_channel = np.max(img_channel,0)
            img_channel = ((img_channel / img_channel.max())*255).astype(np.uint8)
            img.append(img_channel)
        img = np.array(img)
        with open(op.join(annotation_path,pos,'img.pkl'),'wb') as f:
            pickle.dump(img,f)
        
        result = []
        label_nuclei = cv2.imread(op.join(annotation_path,pos,'mask_nuclei.png'),0)
        label_nuclei = resize_label(label_nuclei,(2048,2048))
        label_cell = cv2.imread(op.join(annotation_path,pos,'mask_cell.png'),0)
        label_cell = resize_label(label_cell,(2048,2048))
        cv2.imwrite(op.join(annotation_path,pos,'mask_cell.png'),label_cell)
        cv2.imwrite(op.join(annotation_path,pos,'mask_nuclei.png'),label_nuclei)
        boxes, nb_marker = load_boxes(op.join(annotation_path,pos))
        with open(op.join(annotation_path,pos,'sites.json'), "w") as f:
            json.dump(boxes,f)
        valid_cell = filter_nuclei(label_cell)
        for cell_id in valid_cell:
            result.append(filter_site(boxes[str(cell_id)],cell_id, n = min(nb_marker,2)))
        result = pd.concat(result).fillna(0)
        result.to_csv(op.join(annotation_path,pos,'result.csv'))
        result['position'] = pos
        results.append(result)
    results = pd.concat(results).fillna(0)
    results['exp'] = exp_name
    results.to_csv(op.join(annotation_path,'result.csv'))
        
        
    print('Post Analysis Done and took {}'.format(time.strftime("%H:%M:%S",time.gmtime(time.time()-start))))