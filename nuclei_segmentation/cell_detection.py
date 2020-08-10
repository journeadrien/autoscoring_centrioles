# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 13:21:25 2020

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
import cv2
import utils
import os
import shutil
import json
from skimage.segmentation import random_walker
EXPERIMENT_PATH ="E:\\Adrien\\data\\Experiment"


def resize_img_label(image,label,shape):
    image, _,_,_ = utils.resize_image(image,min_dim=shape[0],max_dim=shape[1],padding=False)
    label, _,_,_ = utils.resize_image(label,min_dim=shape[0],max_dim=shape[1],padding=False)
    return image, label

if __name__ == '__main__':
     # train on the GPU or on the CPU, if a GPU is not available

    datasets_path = glob(EXPERIMENT_PATH+'/*')
    for dataset_path in reversed(datasets_path):
        position = glob(op.join(dataset_path,'images','*',''))
        path_position = {((op.split(op.split(path)[0])[1].split('_')[1][2]),int(op.split(op.split(path)[0])[1].split('_')[2][2])): path for path in position}
        masks_path = [x.replace('images','masks') for x in list(path_position.values())]
        dataset = NucleiExpDataset()
        dataset.load_shapes(path_position.values(),range(29,40),transform = False,color = 'GRAY')
        dataset.prepare()
        predictions = []
        for i, image in enumerate(dataset):
            print(i)
            print(image.shape)
            mask_path = op.join(masks_path[i],'mask_nuclei.png')
            label = cv2.imread(mask_path,0)
            image,label = resize_img_label(image,label,(512,512))
            image = (image*1000).astype(np.uint8) 
            print(image.shape)
            l = random_walker(image,label)
            cv2.imwrite(op.join(masks_path[i],'mask_cell.png'),l)