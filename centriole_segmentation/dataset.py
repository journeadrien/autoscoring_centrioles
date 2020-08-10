# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 23:10:21 2020

@author: journe
"""

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from generator_centriole import GC_2D
import numpy as np
import cv2
import torch
from glob import glob
import os.path as op
import os
from lxml import etree
import math
import random
import utils
from skimage.measure import regionprops



def to_cxcy(xy):
    """
    Convert bounding boxes from boundary coordinates (x_min, y_min, x_max, y_max) to center-size coordinates (c_x, c_y, w, h).
    :param xy: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    """
    return [(xy[0]+xy[2])/2,(xy[1]+xy[3])/2,(xy[2]-xy[0]),(xy[3]-xy[1])]


def to_xy(cxcy):
    """
    Convert bounding boxes from center-size coordinates (c_x, c_y, w, h) to boundary coordinates (x_min, y_min, x_max, y_max).
    :param cxcy: bounding boxes in center-size coordinates, a tensor of size (n_boxes, 4)
    :return: bounding boxes in boundary coordinates, a tensor of size (n_boxes, 4)
    """
    return [cxcy[0]-cxcy[2]/2,cxcy[1]-cxcy[3]/2,cxcy[0]+cxcy[2]/2,cxcy[1]+cxcy[3]/2]

def convert(image,type_ = np.uint16):
    info = np.iinfo(image.dtype) # Get the information of the incoming image type
    data = image.astype(np.float64) / info.max # normalize the data to 0 - 1
    data[data>1] = 1.
    data = np.iinfo(type_).max * data # Now scale by 255
    img = data.astype(type_)
    return img

def div_max(image,background):
    # print(image.max(),40*background)
    max_ = max(image.max(),background)
    if max_ != 0:
        image = image / max_
    image[image < 0] =  0
    image[image > 1]=  1
    return image.astype('float32')


def normalize(image,max_ = None):
    image = (image-image.mean()) / image.std()
    return image

def convert_rgb(img):
    img = img.astype('float32')
    return cv2.cvtColor(img,cv2.COLOR_GRAY2RGB).transpose((2,0,1))

def compute_window(dim_img,nb_img):
    w = math.ceil(dim_img/nb_img)
    if w %2 ==0:
        return w
    else:
        return w-1
    
def div_max_channel(image,background):
    image = np.array([div_max(img,background[i]) for i, img in enumerate(image)])
    return image.astype('float32')

class Image_loader_multiple_channel:
    def __init__(self,path_dataset,path_label,channels,train,sbs):
        self.ia = [Image_loader(path_dataset,path_label,channel,train,sb) for channel,sb in zip(channels,sbs)]
        self.nb_channels = len(channels)
    
    def __getitem__(self,idx):
        return self.ia[random.randint(0,self.nb_channels-1)].__getitem__(0)
            
            

class Image_loader:
    def __init__(self,path_dataset,path_label,channel,train,sb):
        self.sb = sb
        self.img_shape = (400,400)
        self.channel = channel
        self.path_label =path_label
        self.pad_label = self.get_pad_label()
        position = glob(path_dataset+'/*/')
        position.sort()
        print(position)
        self.path_position = {(int(op.split(op.split(path)[0])[1].split('_')[1][2]),int(op.split(op.split(path)[0])[1].split('_')[2][2])): path for path in position}
        self.position = list(self.path_position.keys())[:17]
        print(self.position)
        self.position = self.position[:-1] if train else self.position[-2:]
        self.image = self.load_image()
        self.label, self.box = self.load_label()
        self.img_p = self.get_img_p()
        self.potential_labels = self.get_potential_label()
        self.locations = self.find_locations()
        self.p_pos = self.get_p_pos()
        

    def __getitem__(self, idx):

        image,boxes,labels,background = self.get_image()
        #image = convert_rgb(image)
        boxes = [np.array(to_xy(box)) for box in boxes]
#            mask = np.expand_dims(gc.mask, axis=0)
        image = div_max_channel(image,background)
        return image, boxes, labels
#        return torch.from_numpy(image), torch.FloatTensor(boxes), torch.LongTensor(labels), torch.ByteTensor(difficulties)
        
    def parse_xtml(self,pos):
        xml_tree = etree.parse(op.join(self.path_label,str(pos),'annotation.xml'))
        r = {}
        for image in xml_tree.xpath(r'/annotations/image'):
            points = image.xpath(r'./points')
            img_name = int(image.get("name").split('.')[0])
            r[img_name] = []
            for p in points:
                r[img_name].append([float(x) for x in p.get('points').split(',')])
        return r
        
    def get_potential_label(self):
        potential_labels = {}
        for pos in self.position:
            labels = []
            for l in range(1,5):
                  if np.any(self.img_p[pos] == l):
                        labels.append(l)
            potential_labels[pos] = labels
        return potential_labels
    
    def find_locations(self):
        locations = {pos:{} for pos in self.position}
        for pos in self.position:
            for l in self.potential_labels[pos]:
                loc = np.where(self.img_p[pos] == l)
                locations[pos][l] = {'x':loc[0],'y':loc[1],'len':len(loc[0])}
        return locations
    
    def random_slection(self,pos):
        l = random.choice(self.potential_labels[pos])
        loc = self.locations[pos][l]
        selected_loc = np.random.randint(0,loc['len'])
        x,y = (np.array([loc['x'][selected_loc],loc['y'][selected_loc]])).astype(int)
        return x,y
    
    def get_image(self):
        idx_pos = np.random.choice(list(range(len(self.position))),p=list(self.p_pos.values()))
        
        pos = self.position[idx_pos]
        x,y = self.random_slection(pos)
        label,box = self.find_box(x,y,pos)
        w = int(self.img_shape[0]/2)
        h = int(self.img_shape[1]/2)
        return self.image[pos][...,x-w:x+w,y-h:y+h].copy(),box,label,np.max(self.image[pos],(1,2))/factor


    def find_box(self,x,y,pos):
        boxes =[]
        label = []
        for l,b in zip(self.label[pos],self.box[pos]):
            
            if b[0]<(y+self.img_shape[0]/2) and b[0]>(y-self.img_shape[0]/2) and b[1]<(x+self.img_shape[0]/2) and b[1]>(x-self.img_shape[0]/2):
                box = [(b[0]-y+self.img_shape[0]/2), (b[1]-x+self.img_shape[0]/2), b[2],b[3]]
#                b[0]=b[0]-y+150
#                b[1] = b[1]-x+150
                boxes.append(list(box))
                label.append(int(l))
        return label,boxes

    def get_p_pos(self):
        p_pos = {}
        count = 0
        for pos in self.position:
            p_pos[pos] = len(np.where(self.img_p[pos]>0)[0])
            count = count+p_pos[pos]
        p_pos = {pos:value/count for pos,value in p_pos.items()}
        return p_pos
        
        
    def get_pad_label(self,nb_img = 8):
        img_id = [(ind_x,ind_y) for ind_x in range(nb_img) for ind_y in range(nb_img)]
        target_size = (300,300)
        width_size = compute_window(2048,nb_img)
        length_size = compute_window(2048,nb_img)
        d = {}
        for idx in range(len(img_id)):
            ind_x,ind_y = img_id[idx]
            right = ind_x * width_size
            left =  (ind_x+1) * width_size if ind_x < (nb_img-1) else 2048
            bottom = ind_y * length_size
            top =  (ind_y+1) * length_size if ind_y < (nb_img-1) else 2048
            pad_vertical = int((target_size[0]-(left-right))/2)
            pad_horizontal = int((target_size[1]-(top-bottom))/2)
            d[idx] = [right-pad_vertical,bottom-pad_horizontal]
        return d
        
    def load_image(self,zstack = range(23,43)):
        image = {}
        for pos in self.position:
            path = self.path_position[pos]
            img = utils.get_images(path,self.channel,zstack,'GRAY')
            image[pos] = np.array([np.max(img, axis=0),np.mean(img, axis=0),np.std(img, axis=0)])
        print('Image loaded')
        return image
    
    def load_label(self):
        box = {}
        label = {}
        for pos in self.position:
            label[pos], box[pos] = self.get_box(pos)
        print('Boxes and labels loaded')
        return label,box
            
            
            
    def get_box(self,pos):
        box = []
        label = []
        annotation =  self.parse_xtml(pos)
        for i, value in annotation.items():
            for cy,cx in value:
                append = False


                if self.channel == 'RFP':
                    if cx < 300 and cy < 300:
                        append = True
                        x_pad =0
                        y_pad=0
                elif self.channel == 'GFP':
                    if cy > 300 and cx < 300:
                        append = True
                        y_pad =300
                        x_pad=0
                elif self.channel == 'Cy5':
                    if cx > 300 and cy < 300:
                        append = True
                        x_pad=300
                        y_pad = 0
                if append:

                    cy =cy+self.pad_label[i][1]-y_pad
                    cx = cx+self.pad_label[i][0]-x_pad
                    bbox = np.array([cy,cx,self.sb,self.sb])
                    box.append(bbox)
                    label.append(1)
       
        return label,box
    
    def get_img_p(self):
        p = {}
        for pos in self.position:
            img = np.zeros((2048,2048))
            for b,l in zip(self.box[pos],self.label[pos]):
                y,x = (np.array(b[0:2])).astype(int)
                w = int(self.img_shape[0] /4 -10)
#                print(np.max([0,x-w]),np.min([2048,x+w]),np.max([0,y-w]),np.min([2048,y+w]))
                img[np.max([0,x-w]):np.min([2048,x+w]),np.max([0,y-w]):np.min([2048,y+w])]=l
            # for b,l in zip(self.box[pos],self.label[pos]):
            #     y,x = (np.array(b[0:2])).astype(int)
            #     img[np.max([0,x-w-20]):np.max([0,x-w]),np.max([0,y-w-20]):np.min([2048,y+w+20])]=0
            #     img[np.min([2048,x+w]):np.min([2048,x+w+20]),np.max([0,y-w-20]):np.min([2048,y+w+20])]=0
            #     img[np.max([0,x-w-20]):np.min([2048,x+w+20]),np.max([0,y-w-20]):np.max([0,y-w])]=0
            #     img[np.max([0,x-w-20]):np.min([2048,x+w+20]),np.min([2048,y+w]):np.min([2048,y+w+20])]=0
                
            img[0:int(self.img_shape[0] /2+10),:]=0
            img[:,0:int(self.img_shape[0] /2+10)]=0
            img[2048-int(self.img_shape[0] /2+10):2048,:] = 0
            img[:,2048-int(self.img_shape[0] /2+10):2048]=0
#                img[img<0] = 0
#                img[img>0] = 1
            p[pos]=img
        print('Location computed')
        return p
    


def collate_fn(batch):
    """
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
    This describes how to combine these tensors of different sizes. We use lists.
    Note: this need not be defined in this Class, can be standalone.
    :param batch: an iterable of N sets from __getitem__()
    :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
    """

    images = list()
    targets = list()

    for b in batch:
        images.append(b[0])
        targets.append(b[1])
    #images = torch.stack(images, dim=0)
    return images, targets

class DatasetCentrioleSSD(Dataset):
    def __init__(self, img_id,channel = 'GFP',mode = 'annotate',train=True,sb=8, transform=None):
        """
        Args:
        :param root_dir (string): Directory with all the images
        :param img_id (list): lists of image id
        :param train: if equals true, then read training set, so the output is image, mask and imgId
                      if equals false, then read testing set, so the output is image and imgId
        :param transform (callable, optional): Optional transform to be applied on a sample
        """
        image_path = '/Adrien/data/Experiment/RPE1wt_CEP63+CETN2+PCNT_1/'
        label_path = '/Adrien/data/dataset/centriole'
        self.sb = sb
        self.img_id = img_id
        self.channel = channel
        self.train = train
        self.transform = transform
        self.mode = mode
        if mode == 'annotate':
            self.generator = Image_loader(image_path,label_path,channel,train,self.sb)
        if mode == 'multi_channel':
            self.generator = Image_loader_multiple_channel(image_path,label_path,channel,train,self.sb)


    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, idx):
        if self.mode == 'generator':
            img_shape=(400,400)
            gc = GC_2D(resolution_xy=0.1025,img_shape=img_shape,min_site = 1,max_site=4,channel =self.channel,mode = 6,sb=self.sb)
    
            image = gc.get_data()
            image = convert_rgb(image)
    #            mask = np.expand_dims(gc.mask, axis=0)
            boxes= gc.target
            boxes = [np.array(to_xy(box))*img_shape[0] for box in boxes]
            labels = gc.label
        else:
            image,boxes,labels = self.generator.__getitem__(0)
        return torch.from_numpy(image), {'boxes': torch.FloatTensor(boxes), 'labels':torch.LongTensor(labels)}
    
class DatasetPredictChannel:
    def __init__(self,ir,pos,channel,marker,nuclei_label,cell_label):
        self.t_shape = (400,400)
        self.nuclei_label = nuclei_label
        self.cell_label = cell_label
        self.nb_cells = self.cell_label.max()
        self.image = self.load_image(ir,pos,channel)
        self.background = np.max(self.image,(1,2))/ utils.factor_marker[marker]
        self.i_shape=self.image.shape
        self.centroid = [[int(y) for y in x['centroid']] for x in regionprops(nuclei_label)]
    
    def __len__(self):
        return len(self.nb_cells)

    def __getitem__(self, idx):
        cell_id = idx+1
        x_c,y_c = self.centroid[idx]
        w = int(self.t_shape[0] /2)
        pad_r = max(0,w-x_c)
        pad_l = max(0,(x_c+w)-self.i_shape[-2])
        pad_t = max(0,w-y_c)
        pad_b = max(0,(y_c+w)-self.i_shape[-1])
        xmin = int(max(0,x_c-w))
        ymin = int(max(0,y_c-w))
        xmax = int(min(2048,x_c+w))
        ymax = int(min(2048,y_c+w))
        img = self.image[...,xmin:xmax,ymin:ymax].copy()
        mask = self.cell_label[xmin:xmax,ymin:ymax]
        img[...,mask != cell_id] = 0
        img = np.pad(img,((0,0),(pad_r, pad_l),(pad_t, pad_b)))
        img = div_max_channel(img,self.background)
        return torch.from_numpy(img),(y_c-w,x_c-w)
#        return torch.from_numpy(image), torch.FloatTensor(boxes), torch.LongTensor(labels), torch.ByteTensor(difficulties)
        

        
    def load_image(self,ir,pos,channel):
        img = ir.get_image(pos,channel)

        return np.array([np.max(img, axis=0),np.mean(img, axis=0),np.std(img, axis=0)])

def get_train_valid_loader(epoch_size=100, batch_size=16,channel = 'GFP' ,mode = 'generate',sb=8,split=True,
                           shuffle=False, num_workers=4, val_ratio=0.1, pin_memory=False):

    """Utility function for loading and returning training and validation Dataloader
    :param root_dir: the root directory of data set
    :param batch_size: batch size of training and validation set
    :param split: if split data set to training set and validation set
    :param shuffle: if shuffle the image in training and validation set
    :param num_workers: number of workers loading the data, when using CUDA, set to 1
    :param val_ratio: ratio of validation set size
    :param pin_memory: store data in CPU pin buffer rather than memory. when using CUDA, set to True
    :return:
        if split the data set then returns:
        - train_loader: Dataloader for training
        - valid_loader: Dataloader for validation
        else returns:
        - dataloader: Dataloader of all the data set
    """
    img_id = list(range(epoch_size))
    if split:
        train_id, val_id = train_test_split(img_id, test_size=val_ratio)

        train_transformed_dataset = DatasetCentrioleSSD(img_id=train_id,channel = channel,mode =mode,sb=sb,
                                                   train=True,
                                                   transform=None)
        val_transformed_dataset = DatasetCentrioleSSD(img_id=val_id,channel = channel,mode = mode,sb=sb,
                                                 train=False,
                                                 transform=None)


        train_loader = DataLoader(train_transformed_dataset,batch_size=batch_size,collate_fn=collate_fn,
                                  shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
        val_loader = DataLoader(val_transformed_dataset, batch_size=batch_size,collate_fn=collate_fn,
                                  shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        return (train_loader, val_loader)