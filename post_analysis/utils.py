# -*- coding: utf-8 -*-
"""
Created on Thu May  7 13:08:40 2020

@author: journe
"""
import cv2
from glob import glob
import numpy as np
from skimage.measure import regionprops
import os
import os.path as op
import json
import pandas as pd
import skimage.external.tifffile as tf
import xml.etree.ElementTree as ET
import re

CENTRIOLE_MARKERS = ['CEP63','PCNT_1','CETN2','GTU88']


class Image_Reader:
    def __init__(self,folder_path):
        self.folder_path = folder_path
        self.channels = ['DAPI','GFP','RFP','Cy5']
        self.mode = self.find_mode()
        self.list_file = self.find_file()
        self.z_stack, self.z_res, self.channel_order = self.read_metadata()
        self.pos = self.find_pos()
        self.meta_channel = self.get_marker()
        self.range_z = self.get_z_stack()
    
    def find_mode(self):
        lst_dir = list(os.listdir(self.folder_path))
        if any([x.endswith('.ome.tif') for x in lst_dir]):
            return 'ome'
        return 'folder'
    
    def find_file(self):
        if self.mode == 'ome':
            list_file = os.listdir(self.folder_path)
            return [x for x in list_file if x.endswith('.ome.tif')]
        if self.mode == 'folder':
            return next(os.walk(self.folder_path))[1]
    
    def read_metadata(self):
        if self.mode == 'ome':
            with tf.TiffFile(op.join(self.folder_path,self.list_file[0]),pages =0) as tif:
                z_stack = int(tif.micromanager_metadata['summary']['Slices'])
                z_res = round(float(tif.micromanager_metadata['summary']['z-step_um']),3)
                channel_order_str = tif.micromanager_metadata['summary']['ChNames']
                channel_order = {x.split(' - ')[1]:int(x.split(' - ')[0]) for x in channel_order_str}
        if self.mode == 'folder':
            attrib = self.parse_acquisition()
            z_res = round(float(attrib['acqZstep']),3)
            z_stack = int((float(attrib['acqZtop'])- float(attrib['acqZbottom'])) / z_res)+1
            channel_order_str = [value for key, value in attrib.items() if key.startswith('acqChannelName')]
            channel_order = {x.split(' - ')[1]:int(x.split(' - ')[0]) for x in channel_order_str}
        return z_stack, z_res, channel_order
    
    
    def parse_acquisition(self):
        tree =  ET.parse(glob(op.join(self.folder_path,'*_Acquisitio-Settings.txt'))[0])
        d ={}
        for p in tree.getiterator():
            if p.tag == 'entry':
                d[p.attrib['key']] = p.attrib['value']
        return d           
    
    def find_pos(self):
        pos ={}
        for file in self.list_file:
            pos_file = re.findall('\d{3}_\d{3}',file)[-1]
            x,y = pos_file.split('_')
            pos[pos_file] = {'x': int(x), 'y': int(y), 'file_path':op.join(self.folder_path,file)}
        return pos
    
    def get_z_stack(self):
        range_z = 66
        focus = int(self.z_stack/2)
        start = max(0,int(focus - range_z*self.z_res/2))
        stop = min(self.z_stack,int(focus+2 + range_z*self.z_res/2))
        return np.array(range(start,stop))
    
    def get_image(self,pos,channel):
        path_file = self.pos[pos]['file_path']
        if self.mode == 'ome':
            index_ch = self.meta_channel[channel]['index']
            pages = list((index_ch-1) * self.z_stack + self.range_z)
            with tf.TiffFile(op.join(self.folder_path,path_file),pages = pages) as tif:
                img = tif.asarray()
        if self.mode == 'folder':
            path_list = [self.get_path_image(path_file,channel,z) for z in self.range_z]
            img =[]
            for file in path_list:
                with tf.TiffFile(file) as tif:
                    img.append(tif.asarray())
            img = np.array(img)
        return img
            
              
    def get_path_image(self,path,channel,z):
        z_str = ('' if z>99 else '0' if z>9 else '00') + str(z)
        return glob(op.join(path,'*'+channel+'*'+str(z_str)+'*.tif'))[0]
    
    def get_marker(self):
        _,folder_name = op.split(self.folder_path)
        info_exp = folder_name.split('_')
        if len(info_exp) == 4:
            cell_line, treatment, markers, nb_exp = info_exp
        elif len(info_exp) == 3:
            cell_line, markers, nb_exp = info_exp
        else:
            raise 'Format Exp name not recognise'
        markers = markers.split('+')
        if 'DAPI' not in markers:
            markers.insert(0,'DAPI')
        meta_channel = {}
        for ch, mar in zip(self.channels[:len(markers)], markers):
            meta_channel[ch] = {'name':mar,'index':self.channel_order[ch]}

        return meta_channel
def load_img(fname, color='RGB'):
    img = cv2.imread(fname,cv2.IMREAD_ANYDEPTH)
    if color == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img  

def get_image(path,channel,z_stack,color):
    z_stack = ('' if z_stack>99 else '0' if z_stack>9 else '00') + str(z_stack)
    return load_img(glob(path+'*'+channel+'*'+str(z_stack)+'*.tif')[0], color)

def get_images(path,channels,z_stacks,color):
    if isinstance(channels, str):
        return np.array([get_image(path,channels,z_stack,color) for z_stack in z_stacks])
    if isinstance(z_stacks, int):
        return {channel : get_image(path,channel,z_stacks) for channel in channels}
    return {channel : np.array([get_image(path,channel,z_stack,color) for z_stack in z_stacks]) for channel in channels}

def resize_img_label(image,label,shape):
    image, _,_,_ = resize_image(image,min_dim=shape[0],max_dim=shape[1],padding=False)
    label, _,_,_ = resize_image(label,min_dim=shape[0],max_dim=shape[1],padding=False)
    return image, label


def resize_label(label,shape):
    nb_label = label.max()
    labels = []
    for l in range(1,nb_label+1):
        label_resize = resize_image2((label == l).astype(np.uint8),shape)
        label_resize = (label_resize>0.5).astype(np.uint8)
        labels.append(label_resize * l)
    return np.max(labels,0)

def resize_image2(image, shape):
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    scale = 1
    min_dim = min(shape)
    max_dim = max(shape)
    # Scale?
    if min_dim:
        # Scale up but not down
        scale = max(1, min_dim / min(h, w))
    # Does it exceed max dim?
    if max_dim:
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max
    # Resize image and mask
    if scale != 1:
        image = cv2.resize(
            image, (round(w * scale),round(h * scale)),interpolation=cv2.INTER_CUBIC)
    return image

def load_boxes(path,threshold = 0.8):
    markers_path = [x for x in os.listdir(path) if (x.endswith('.json') and x not in ['nuclei.json','sites.json'])]
    marker_name = [x.split('.')[0] for x in markers_path]
    dict_box = {}
    for marker, marker_path in zip(marker_name,markers_path):
        if not os.path.isfile(op.join(path, marker_path)):
            continue
        with open(op.join(path,marker+'.json')) as f:
            d = json.load(f)
        for cell_id, data in d.items():
            if cell_id in dict_box:
                dict_box[cell_id].update({marker:select_boxes(data['boxes'],data['scores'],threshold)})
            else:
                dict_box[cell_id] = {marker:select_boxes(data['boxes'],data['scores'],threshold)}
    boxes = {cell_id : find_site(cell_boxes) for cell_id,cell_boxes in dict_box.items()}
    return boxes, len(marker_name)

def select_boxes(boxes,scores,threshold):
    return np.array(boxes)[np.array(scores)>threshold]

def find_max_box(box,box2):
    box[0] = min(box[0],box2[0])
    box[1] = min(box[1],box2[1])
    box[2] = max(box[2],box2[2])
    box[3] = max(box[3],box2[3])
    return box

def find_site(boxes_dict):
    sites = []
    for marker, boxes in boxes_dict.items():
        for box in boxes:
            if box.size ==0:
                continue
            belong_to_no_site = True
            for site in sites:
                if np.linalg.norm([site['box'][:2]-box[:2]]) <20:
                    belong_to_no_site = False
                    site['box'] = list(find_max_box(site['box'],box))
                    if marker in site['label']:
                        site['label'][marker] = site['label'][marker] + 1
                    else:
                        site['label'][marker] = 1
                    break
            if belong_to_no_site:
                sites.append({'box':list(box),'label':{marker:1}})
                    
            if not sites:
                sites = [{'box':list(box),'label':{marker:1}}]
    return sites

def filter_site(sites,cell_id,n =2):
    valid_sites = [site['label'] for site in sites if len(site['label'])>=n]
    nb_sites = len(valid_sites)
    unique_marker = list(np.unique(flat_list([site.keys() for site in valid_sites])))
    result = pd.DataFrame(data = 0,columns = ['nb sites']+ unique_marker,index=[cell_id])
    for site in valid_sites:
        for marker, nb_marker in site.items():
            result[marker] = result[marker] + nb_marker
    result['nb sites'] = nb_sites
    return result
    
def flat_list(lst):
    return [item for sublist in lst for item in sublist]

def filter_nuclei(label):
    centroid = [[int(y) for y in x['centroid']] for x in regionprops(label)]
    valid_cell = []
    for i, (x_c,y_c) in enumerate(centroid):
        w = 120
        if x_c < w or y_c < w or x_c>2048-w or y_c>2048-w:
            continue
        valid_cell.append(i)
    return valid_cell
    