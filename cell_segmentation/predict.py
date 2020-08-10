# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 13:21:25 2020

@author: journe
"""


from glob import glob
import torch
import os.path as op
import numpy as np
from skimage import io
import cv2
import sys
import os
import utils
import time
from scipy.ndimage import distance_transform_edt
EXPERIMENT_PATH ="E:\\Adrien\\data\\Experiment"


if __name__ == '__main__':
    print('Starting Cell Segmentation')
    start = time.time()
    shape = (512,512)
    
    # train on the GPU or on the CPU, if a GPU is not available
    data_path = sys.argv[1]
    annotation_path = sys.argv[2]

    for pos in next(os.walk(annotation_path))[1]:
        label = cv2.imread(op.join(annotation_path,pos,'mask_nuclei.png'),0)
        threshold_d=200*label.shape[0]/2048
        # image,label = utils.resize_img_label(image,label,shape)
        
        
        distance,(i,j) = distance_transform_edt(label == 0, return_indices = True)
        cell_label = np.zeros(label.shape,np.uint8)
        dilate_mask = distance<threshold_d
        cell_label[dilate_mask] = label[i[dilate_mask],j[dilate_mask]]
        cv2.imwrite(op.join(annotation_path,pos,'mask_cell.png'),cell_label)
        
    print('Cell Segmentation Done and took {}'.format(time.strftime("%H:%M:%S",time.gmtime(time.time()-start))))