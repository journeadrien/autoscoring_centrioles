# -*- coding: utf-8 -*-
"""
Created on Sun May  3 19:16:21 2020

@author: journe
"""
import numpy as np
import os
import utils
import os.path as op
import cv2
from skimage.io import imsave
from tqdm import tqdm
from skimage.measure import regionprops
import matplotlib.pyplot as plt


factor = {'DAPI':1,
          'GFP':1.5,
          'RFP':1.5,
          'Cy5':1.5}

def transform(image,channel,max_):
    image = (image.astype('float32')/max_[channel])*factor[channel]
    image[image>1.] = 1.
    image[image<0.] = 0.
    return image.astype('float32')
    
def transform1(image,mean):
    image = image.astype('float32')  / mean
    return image.astype('float32')


def div_max(image):

    max_ = max(image.max(),1*image.mean())
    if max_ != 0:
        image = image / max_
    image[image < 0] =  0
    image[image > 1]=  1
    return image.astype('float32')

    
def div_max_channel(image):
    for i in range(image.shape[-1]):
        image[:,:,i] = image[:,:,i] / image[:,:,i].max()
    return image.astype('float32')

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

class DatasetAnnotation:
    def __init__(self,image,nuclei_label,cell_label):
        self.t_shape = (300,300)
        self.nuclei_label = nuclei_label
        self.cell_label = cell_label
        self.nb_cells = self.cell_label.max()
        self.image = image
        self.i_shape=self.image.shape
        self.centroid = [[int(y) for y in x['centroid']] for x in regionprops(nuclei_label)]
    
    def __len__(self):
        return self.nb_cells

    def __getitem__(self, idx):
        cell_id = idx+1
        x_c,y_c = self.centroid[idx]
        w = 120
        if x_c < w or y_c < w or x_c>2048-w or y_c>2048-w:
            return None
        w = int(self.t_shape[0] /2)
        pad_r = max(0,w-x_c)
        pad_l = max(0,(x_c+w)-self.i_shape[0])
        pad_t = max(0,w-y_c)
        pad_b = max(0,(y_c+w)-self.i_shape[1])
        xmin = int(max(0,x_c-w))
        ymin = int(max(0,y_c-w))
        xmax = int(min(2048,x_c+w))
        ymax = int(min(2048,y_c+w))
        img = self.image[xmin:xmax,ymin:ymax,...].copy()
        mask = self.cell_label[xmin:xmax,ymin:ymax]

        img[mask != cell_id,...] = 0

        img = np.pad(img,((pad_r, pad_l),(pad_t, pad_b),(0,0)))
        return img
#        return torch.from_numpy(image), torch.FloatTensor(boxes), torch.LongTensor(labels), torch.ByteTensor(difficulties)
class Dataset:
    def __init__(self,image,nuclei_label,cell_label):
        self.t_shape = (300,300)
        self.nuclei_label = nuclei_label
        self.cell_label = cell_label
        self.nb_cells = self.cell_label.max()
        print(image[nuclei_label == 0,...].mean())
        self.image = image
        self.i_shape=self.image.shape
        self.centroid = [[int(y) for y in x['centroid']] for x in regionprops(nuclei_label)]
    
    def __len__(self):
        return self.nb_cells

    def __getitem__(self, idx):
        cell_id = idx+1
        x_c,y_c = self.centroid[idx]
        w = 120
        if x_c < w or y_c < w or x_c>2048-w or y_c>2048-w:
            return None
        w = int(self.t_shape[0] /2)
        pad_r = max(0,w-x_c)
        pad_l = max(0,(x_c+w)-self.i_shape[0])
        pad_t = max(0,w-y_c)
        pad_b = max(0,(y_c+w)-self.i_shape[1])
        xmin = int(max(0,x_c-w))
        ymin = int(max(0,y_c-w))
        xmax = int(min(2048,x_c+w))
        ymax = int(min(2048,y_c+w))
        img = self.image[xmin:xmax,ymin:ymax,...].copy()
        mask = self.cell_label[xmin:xmax,ymin:ymax]

        img[mask != cell_id,...] = 0

        img = np.pad(img,((pad_r, pad_l),(pad_t, pad_b),(0,0)))
        return (img * (2**8-1)).astype(np.uint8)
    
    
class ConcatDataset:
    def __init__(self,datasets):
       self.datasets = datasets
      
    
    def __len__(self):
        return self.datasets[0].__len__()
    
    def __getitem__(self,idx):
        i = [dataset.__getitem__(idx) for  dataset in self.datasets]
        if i[0] is None:
            return None
        return (np.concatenate([np.concatenate(i[:2],1),np.concatenate(i[2:],1)])*(2**8-1)).astype(np.uint8)
    # def load_image(self,path_image,zstack = range(18,50)):
    #     img_DAPI = utils.get_images(path_image,'DAPI',zstack,'GRAY')
    #     img_RFP = utils.get_images(path_image,'RFP',zstack,'GRAY')
    #     img_GFP = utils.get_images(path_image,'GFP',zstack,'GRAY')
    #     img_Cy5 = utils.get_images(path_image,'Cy5',zstack,'GRAY')
    #     print(img_DAPI.dtype)
    #     return np.array([np.max(img_DAPI, axis=0),np.std(img_DAPI, axis=0),np.max(img_RFP,axis = 0),np.std(img_RFP, axis=0)]).astype(np.uint16)
if __name__ == '__main__':
    j = 0
    mode = 'training'
    print('Starting Centriole Detection')
    data_path="E:\\Adrien\\data\\Experiment\\RPE1wt_CEP63+Cyclin-PCNA+PCNT_1"
    annotation_path="E:\\Adrien\\data\\Annotation\\RPE1wt_CEP63+Cyclin-PCNA+PCNT_1" 
    cycle_phase_folder = "E:\\Adrien\\data\\dataset\\cycle_phase" 
    classes_lst = ['G1','S','G2','M','bad']
    # with open(op.join(cycle_phase_folder,'classes.txt'),'w+') as f:
    #     f.write('\n'.join(classes_lst))
    ir = utils.Image_Reader(data_path)
    channels = ['DAPI'] if mode == 'training' else ir.meta_channel.keys()
    max_image = {channel : {} for channel in channels}
    std_image =  {channel : {} for channel in channels}
    mean_image =  {channel : {} for channel in channels}
    label_nuclei_dict = {}
    label_cell_dict = {}
    positions =list(ir.pos.keys())
    mean_background = {}
    for pos in tqdm(positions):
        label_nuclei = cv2.imread(op.join(annotation_path,pos,'mask_nuclei.png'),0)
        label_nuclei = resize_label(label_nuclei,(2048,2048))
        label_cell = cv2.imread(op.join(annotation_path,pos,'mask_cell.png'),0)
        label_cell = resize_label(label_cell,(2048,2048))
        label_nuclei_dict[pos] = label_nuclei
        label_cell_dict[pos] = label_cell
        for channel in channels:
            img_channel = ir.get_image(pos,channel)
            max_image[channel][pos] = np.max(img_channel,0)
            std_image[channel][pos] = np.std(img_channel,0)
            mean_image[channel][pos] = np.mean(img_channel,0)
            if channel == 'DAPI':
                mean_background[pos] = img_channel[:,label_nuclei == 0].mean()
    max_image = {channel : {pos: transform1(x,mean_background[pos]) for pos, x in images.items()} for channel, images in max_image.items()}
    std_image = {channel : {pos: transform1(x,mean_background[pos]) for pos, x in images.items()} for channel, images in std_image.items()}
    mean_image = {channel : {pos: transform1(x,mean_background[pos]) for pos, x in images.items()} for channel, images in mean_image.items()}
    
    max_max = {channel: np.max(list(x.values())) for channel, x in max_image.items()}
    max_std = {channel: np.max(list(x.values())) for channel, x in std_image.items()}  
    max_mean = {channel: np.max(list(x.values())) for channel, x in mean_image.items()}  
    max_image = {channel : {pos: transform(x,channel,max_max) for pos, x in images.items()} for channel, images in max_image.items()}
    std_image = {channel : {pos: transform(x,channel,max_std) for pos, x in images.items()} for channel, images in std_image.items()}
    mean_image = {channel : {pos: transform(x,channel,max_mean) for pos, x in images.items()} for channel, images in mean_image.items()}
    for pos in tqdm(positions):
        label_nuclei = label_nuclei_dict[pos]
        label_cell = label_cell_dict[pos]
        if mode == 'training':
            concatdataset = Dataset(np.moveaxis(np.array([max_image['DAPI'][pos],mean_image['DAPI'][pos],std_image['DAPI'][pos]]),0,-1),label_nuclei,label_cell)
        else:
            d1 = DatasetAnnotation(np.stack([max_image['DAPI'][pos]]*3,axis = -1),label_nuclei,label_cell)
            d2 = DatasetAnnotation(np.stack([std_image['DAPI'][pos]]*3,axis = -1),label_nuclei,label_cell)
            d3 = DatasetAnnotation(np.moveaxis(np.array([max_image[channel][pos] for channel in ['GFP','RFP','Cy5']]),0,-1),label_nuclei,label_cell)
            d4 = DatasetAnnotation(np.moveaxis(np.array([std_image[channel][pos] for channel in ['GFP','RFP','Cy5']]),0,-1),label_nuclei,label_cell)
            concatdataset = ConcatDataset([d1,d2,d3,d4])
        for image in concatdataset:
            if image is None:
                continue
            if mode == 'training':
                imsave(op.join(cycle_phase_folder,str(j)+'.png'),image) 
            else:
                cv2.imwrite(op.join(cycle_phase_folder,str(j)+'.png'),image) 
            # with open(op.join(cycle_phase_folder,str(j)+'.txt'),'w+') as f:
            #     f.write('0 0. 0. 0.01 0.01')
            j = j+1