# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 10:17:12 2020

@author: journe
"""

import matplotlib.pyplot as plt
from dataset import NucleiExpDataset
from engine import evaluate
from model import get_model_instance_segmentation
from glob import glob
import torch
import os.path as op
import numpy as np
from skimage import io
from skimage import measure
from skimage.morphology import closing, square
import cv2
import sys
import utils
import os
import shutil
import json
import time
from tqdm import tqdm
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def keep_biggest(mask):
    mask_label,nb_label = measure.label(mask,return_num = True)
    size_object = np.array([(mask_label == i).sum() for i in range(1,nb_label+1)])
    return (mask_label == (size_object.argmax() +1)).astype(np.uint8)

def collate_fn(batch):
    return list(batch)
def resize_concat_masks(masks,shape):
    labels = []
    for i in range(masks.shape[0]):
        mask,_,_,_ = utils.resize_image(masks[i],min_dim=shape[0],max_dim=shape[1],padding=False)
        mask = (mask>0.5).astype(np.uint8)
        mask = closing(mask, square(4))
        mask = keep_biggest(mask)
        labels.append(mask*(i+1))
    return np.array(labels).max(0).astype(np.uint8)
    
def save_result(masks_path,prediction):
    boxes = prediction['boxes'].cpu().numpy()
    masks = prediction['masks'].cpu().numpy().squeeze(1)
    masks = resize_concat_masks(masks,shape = (512,512))
    scores = prediction['scores'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    if op.exists(masks_path):
        shutil.rmtree(masks_path)
    os.makedirs(masks_path, 0o777)
    annotations = []
    file_name = 'mask_nuclei'
    cv2.imwrite(op.join(masks_path, file_name+'.png'), masks)
    for i in range(len(labels)):
        
        annotations.append({
                "box": list(np.array(boxes[i]).astype(float)),
                "score": scores[i].astype(float),
                "label": int(labels[i]),
                "mask": file_name + '.png'
            })

        # print(prediction)
        with open(op.join(masks_path, 'nuclei.json'), 'w') as fp:
            json.dump(annotations, fp)



if __name__ == '__main__':
    print('Starting Nucleus Segmentation')
    start = time.time()
    data_path = sys.argv[1]
    annotation_path = sys.argv[2]
     # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # our dataset has two classes only - background and person
    num_classes = 2

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)
    
    # move model to the right device
    model.to(device)
    
    model.load_state_dict(torch.load(op.join(ROOT_DIR,'model','0.pt')))

    ir = utils.Image_Reader(data_path)
   
    dataset = NucleiExpDataset()
    dataset.load_shapes(ir)
    dataset.prepare()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,collate_fn=collate_fn)
    model.eval()
    predictions = []
    for i, images in enumerate(data_loader):
        c=images
        with torch.no_grad():
            prediction = model([image.to(device) for image in images])
        predictions.extend(prediction)
    for pos,prediction in zip(ir.pos.keys(),predictions):
        save_result(op.join(annotation_path,pos), prediction)
    print('Nucleus Segmentation Done and took {}'.format(time.strftime("%H:%M:%S",time.gmtime(time.time()-start))))