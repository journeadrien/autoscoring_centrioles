# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 23:34:13 2020

@author: journe
"""

from dataset import DatasetPredictChannel
from engine import evaluate
from model import get_model_mobile_net,get_model_resnet
from glob import glob
import torch
import os.path as op
import numpy as np
from skimage import io
import cv2
import utils
import os
import shutil
import json
from skimage.segmentation import random_walker
from utils import resize_label,read_marker_txt, size_marker, Image_Reader
import json
import sys
import time 
from tqdm import tqdm
EXPERIMENT_PATH ="E:\\Adrien\\data\\Experiment"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def remove_json_file(path):
    markers_path = [x for x in os.listdir(path) if (x.endswith('.json') and x not in ['sites.json','nuclei.json'])]
    for marker in markers_path:
        os.remove(op.join(path,marker))
    
    
def load_model(size,device):
     
     model = get_model_resnet(last_channel = 64,sb=(int(size/2)), box_score_thresh = 0.5,box_nms_thresh = 0.2)
     model.to(device)
     model.load_state_dict(torch.load(op.join(ROOT_DIR,'model','resnet_'+str(size)+'.pt')))
     return model
if __name__ == '__main__':
    print('Starting Centriole Detection')
    start = time.time()
    data_path = sys.argv[1]
    annotation_path = sys.argv[2]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
     # train on the GPU or on the CPU, if a GPU is not available
    ir = Image_Reader(data_path)
    marker_channel ={}
    for channel, value in ir.meta_channel.items():
        if value['name'] in size_marker.keys():
            marker_channel[value['name']] = channel
    labels_cell = []
    labels_nuclei = []
    for pos in tqdm(list(ir.pos.keys()),'Load label'):
        remove_json_file(op.join(annotation_path,pos))
        nuclei_path = op.join(annotation_path,pos,'mask_nuclei.png')
        cell_path = op.join(annotation_path,pos,'mask_cell.png')
        label_nuclei = cv2.imread(nuclei_path,0)
        label_nuclei = resize_label(label_nuclei,(2048,2048))
        label_cell = cv2.imread(cell_path,0)
        label_cell = resize_label(label_cell,(2048,2048))
        labels_cell.append(label_cell)
        labels_nuclei.append(label_nuclei)
    for marker,channel in tqdm(marker_channel.items(),'By Marker'):
        model = load_model(size_marker[marker],device)
        for i,pos in enumerate(ir.pos.keys()):
            ia = DatasetPredictChannel(ir,pos,channel, marker,labels_nuclei[i],labels_cell[i])
            model.eval()
            predictions = {}
            for cell_id, (image, pad) in enumerate(ia):
                 with torch.no_grad():
                     prediction = model([image.to(device)])[0]
                     prediction["scores"] = prediction["scores"].tolist()
                     prediction["labels"] = prediction["labels"].tolist()
                     prediction["boxes"] = prediction["boxes"].tolist()
                     for bbox in prediction["boxes"]:
                         bbox[0] = bbox[0] + pad[0]
                         bbox[1] = bbox[1] + pad[1]
                         bbox[2] = bbox[2] + pad[0]
                         bbox[3] = bbox[3] + pad[1]
                     prediction['pad'] = pad
                 predictions[cell_id] = prediction
            with open(op.join(annotation_path,pos,marker+'.json'), "w") as f:
                json.dump(predictions, f)
    print('Centriole Detection Done and took {}'.format(time.strftime("%H:%M:%S",time.gmtime(time.time()-start))))
 
                    