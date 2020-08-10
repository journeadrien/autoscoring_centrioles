# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 23:09:08 2020

@author: journe
"""
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2
from skimage.filters import gaussian
from matplotlib.patches import Circle
import imageio as io
from scipy import ndimage

DICT_CHANNEL ={'RFP':{'mean':[1200,1200],'p_art':[0.3,0.7],'mean_art':[2000,2600],'std':[50,100],'sigma':[0.16,0.27],'nb_centriole':4},
               'GFP':{'mean':[2600,2600],'p_art':[0.05,0.4],'mean_art':[1000,1600],'std':[80,120],'sigma':[0.15,0.25],'nb_centriole':2},
               'Cy5':{'mean':[2000,2000],'std':[100,200],'p_art':[0.1,0.7],'mean_art':[1500,2000],'sigma':[0.2,0.4],'nb_centriole':2}}
 
COLOR_LABEL = {1:(0,1,0),2:(0,0,1),3:(0,1,1),4:(1,1,0)}


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
    
class GC_2D:
    def __init__(self,resolution_xy,img_shape,min_site,max_site,channel,info = False,mode = 1,sb = 8,model = 'Box',how = 'cxcy'):
        self.prop =DICT_CHANNEL[channel]
        self.img_shape = img_shape
        self.sb = sb
        self.PSNR = random.uniform(1,8)
        
#        self.PSNR = np.random.choice([1,1.5,2,2.5,3], p=[0.3, 0.25, 0.15, 0.1, 0.2])+random.uniform(0,0.5)
        self.mean = np.min([50000,random.uniform(self.prop['mean'][0],self.prop['mean'][1])])
        self.std = random.uniform(self.prop['std'][0],self.prop['std'][1])
        self.img = np.random.normal(self.mean, self.std, self.img_shape)
        self.PSNR_threshold = 15
        
        self.img = self.img.astype('float32')
        self.resolution_xy = resolution_xy
        self.dist_min = 7
        self.x_vector = list(np.arange(0,self.img_shape[0]*resolution_xy,resolution_xy))
        self.y_vector = list(np.arange(0,self.img_shape[1]*resolution_xy,resolution_xy))
        self.padding = 5
        self.sigma_sigma =1
        self.mean_intensity = 1.5*2**15
        self.sigma_image =  random.uniform(self.prop['sigma'][0],self.prop['sigma'][1])
        self.minimun_intensity = self.mean 
        self.nb_sites =random.randint(min_site, max_site)
        self.nb_sites_art =random.randint(0,2)

        self.nb_centrioles = { site : np.random.randint(1,self.prop['nb_centriole']+1) for site in range(self.nb_sites)}
        

        self.X, self.Y = self.create_coordinate_1(self.nb_sites,self.dist_min)
        self.intensity = [2**16/self.PSNR if i == 0 else self.law_intensity() for i in range(len(self.X))]
        self.sigma_x = [self.law_sigma() for _ in self.X]
        self.sigma_y = [2*self.sigma_image - self.sigma_x[i] for i in range(len(self.X))]
        self.centrioles = {site :{1: {'X':self.X[site],'Y':self.Y[site],'i':self.intensity[site], 'sigma_x':self.sigma_x[site],'sigma_y':self.sigma_y[site]}} for site in range(self.nb_sites)}
        self.add_artifact()
        self.create_coordinate_2()

        self.img_gaussian_art = self.add_gaussian_art()
        self.img_gaussian_art = gaussian(self.img_gaussian_art*12,6)+self.img_gaussian_art
        self.img = self.img +self.img_gaussian_art
        self.img = gaussian(self.img,1)
        self.img[self.img<0] = 0

        self.img_gaussian = self.add_gaussian()
        self.img = self.img + self.img_gaussian
        
        self.target = self.create_mask() if model == 'mask' else self.box_label(mode = mode)
        if info:
            self.info()
            self.visualize(how)
            

            
    def law_intensity(self,art = False):
        if art == False:
            while True:
                i=(self.mean_intensity*np.random.normal(1,0.13,1)[0])/self.PSNR
                if  i>0 and i<(np.iinfo(np.uint16).max)/self.PSNR:
                    return i 
        else:
            return random.uniform(self.mean + self.std, self.minimun_intensity/3)
    
    def law_sigma(self):
         while True:
            sigma=self.sigma_image*np.random.normal(1,0.13,1)[0]
            if sigma > self.sigma_image-1 and sigma< self.sigma_image+1:
                return sigma
    
    def visualize(self,how):
        fig, ax = plt.subplots(figsize=(18, 18))
        ax.imshow(self.img, cmap=plt.get_cmap('gray'))
        plt.show()
        print(self.target)
        print(self.label)
        fig, ax = plt.subplots(figsize=(18, 18))
        ax.imshow(self.add_boxes(how))
        plt.show()

    def normalize(self, img):
        info = info = np.iinfo(np.uint16) # Get the information of the incoming image type
        data = img / info.max # normalize the data to 0 - 1
        data[data>1] = 1.
        return data
     
    def add_boxes(self,how):
        
        img_box = np.zeros(self.img_shape+(3,))
        img_box[:,:,0] = div_max(normalize(self.img))
        if how == 'xy':
            for x1,y1,x2,y2 in self.target:
                cv2.rectangle(img_box, (np.float32(x1*self.img_shape[0]),np.float32(y1*self.img_shape[0])), (np.float32(x2*self.img_shape[0]), np.float32(y2*self.img_shape[0])), (0,1,0), 1)
        if how == 'cxcy':
            for i,(x,y,w,h) in enumerate(self.target):
                x1,y1,x2,y2 = to_xy([x,y,w,h])
                c = COLOR_LABEL[self.label[i]]
                cv2.rectangle(img_box, (np.float32(x1*self.img_shape[0]),np.float32(y1*self.img_shape[0])), (np.float32(x2*self.img_shape[0]), np.float32(y2*self.img_shape[0])), c, 1)
        return img_box
    
    def add_circles(self):
        img_box = np.zeros(self.img_shape+(3,))
        img_box[:,:,2] = self.normalize(self.img)
        img_box[:,:,0] = self.normalize(self.img)
        for x,y,r in self.target:
            cv2.circle(img_box,(x, y), int(r), (0,1,0), 1)
            
        return img_box

    def distance_check(self,x_new,y_new,X_old,Y_old,threshold):
        dist = [np.linalg.norm([x_new-x,y_new-y])>threshold for x,y in zip(X_old,Y_old)]
        return all(dist)

    def info(self):
        print(self.centrioles)
        print('sigma: '+str(self.sigma_image))
        print('std_img : '+str(self.std))
        print('mean_img : '+str(self.mean))
        print('nb_site : '+str(self.nb_sites))
        print('p_cell: '+ str(self.p_cell))
        print('signal to noise: '+ str(self.PSNR))
        
        
    def box_label(self,mode = 1):
        target = []
        self.label = []
        for key,value in self.centrioles.items():
            x1 = []
            x2 = []
            y1 = []
            y2 = []
            for key, coor in value.items(): 
            
                x = coor['X']
                y = coor ['Y']
                sigma = np.mean([coor['sigma_x'],coor['sigma_y']])
#                if key ==2:
#                    d = coor['d']
#                    d_min = np.sqrt(2*sigma*sigma*np.max([0,np.log(3*i/coor['i'])]))
                i=coor['i']
                r = np.sqrt(2*sigma*sigma*np.max([0,np.log(i)]))/1.3
                y1_vec = np.max([0,x-r])
                y2_vec = np.min([self.x_vector[-1],x+r])
                x1_vec = np.max([0,y-r])
                x2_vec = np.min([self.y_vector[-1],y+r])
                x1.append(self.x_vector.index(min(self.x_vector, key=lambda a:abs(a-x1_vec)))/self.img_shape[0])
                x2.append(self.x_vector.index(min(self.x_vector, key=lambda a:abs(a-x2_vec)))/self.img_shape[0])
                y1.append(self.y_vector.index(min(self.y_vector, key=lambda a:abs(a-y1_vec)))/self.img_shape[1])
                y2.append(self.y_vector.index(min(self.y_vector, key=lambda a:abs(a-y2_vec)))/self.img_shape[1])
                if mode ==6:
                    h = (self.resolution_xy * self.sb/2)
                    w = (self.resolution_xy * self.sb/2)
                    target.append(to_cxcy(np.array([y-w,x-h,y+w,x+h])/(self.img_shape[0]*self.resolution_xy)))
                    self.label.append(1)
                if (mode == 1  or mode ==3):
                    target.append(to_cxcy([x1[-1],y1[-1],x2[-1],y2[-1]]))
                    if self.PSNR  < self.PSNR_threshold:
                        self.label.append(1)
                    
            if (mode ==2):
                target.append(to_cxcy([np.min(x1),np.min(y1),np.max(x2),np.max(y2)]))
                if self.PSNR <self.PSNR_threshold:
                    
                    if len(x1) ==1:
                        self.label.append(1)
                    else:
#                        print(d_min,d)
#                        if d<d_min:
#                            self.label.append(1)
#                        else:
                        self.label.append(int(len(x1)))
                        
                else:
 
                    self.label.append(3)
            if mode ==4:
                target.append(to_cxcy([np.min(x1),np.min(y1),np.max(x2),np.max(y2)]))

                    
                       
                        
            if mode == 5:
                
                if len(x1) ==1:
                    target.append(to_cxcy([np.min(x1),np.min(y1),np.max(x2),np.max(y2)]))
                    self.label.append(1)
                else:
                    if 2.1*d<r or self.PSNR>self.PSNR_threshold:
                        target.append(to_cxcy([x1[0],y1[0],x2[0],y2[0]]))
                        self.label.append(1)
                    else:
                        target.append(to_cxcy([np.min(x1),np.min(y1),np.max(x2),np.max(y2)]))
                        self.label.append(2)
                
#        for key,value in self.art.items():
#            x = []
#            y = []
#            for _, coor in value.items(): 
#                x.append(coor['X'])
#                y.append(coor['Y'])
#            self.label.append(3)
#            y1_vec = np.min(x)-0.15
#            y2_vec = np.max(x)-0.15
#            x1_vec =np.min(y)+0.15
#            x2_vec = np.max(y)+0.15
#            x1 = self.x_vector.index(min(self.x_vector, key=lambda a:abs(a-x1_vec)))/self.img_shape[0]
#            x2 = self.x_vector.index(min(self.x_vector, key=lambda a:abs(a-x2_vec)))/self.img_shape[0]
#            y1= self.y_vector.index(min(self.y_vector, key=lambda a:abs(a-y1_vec)))/self.img_shape[1]
#            y2 = self.y_vector.index(min(self.y_vector, key=lambda a:abs(a-y2_vec)))/self.img_shape[1]
#            target.append(to_cxcy([x1,y1,x2,y2]))
        return target
            
      
    def add_artifact(self):
        self.X_art=[]
        self.Y_art = []
        self.intensity_art = []
        self.sigma_x_art = []
        self.sigma_y_art = []
        def define_p(x_vec,y_vec,X,Y,p):
            r = []
            for x,y in zip(X,Y):
                x = x + random.uniform(-50,50)*self.resolution_xy
                y = y + random.uniform(-50,50)*self.resolution_xy
                x_ = x_vec-x/self.resolution_xy
                y_ = y_vec-y/self.resolution_xy
                x, y = np.meshgrid(x_, y_) 
                
                img = x**2/np.random.uniform(1,5) +y**2/np.random.uniform(1,5)
                y_center = abs(y_).argmin()
                x_center = abs(x_).argmin()
                img[y_center-1:y_center+1,x_center-1:x_center+1] = 10000
                r.append(img)
            return p * (np.min(r,0)<random.uniform(3000,5000)) +0.01
        x_w = np.random.randint(6,8)
        y_w = np.random.randint(6,8)
        x_vec = np.arange(0,self.img_shape[0],x_w)
        y_vec = np.arange(0,self.img_shape[1],y_w)
        self.p_cell = random.uniform(self.prop['p_art'][0],self.prop['p_art'][1])
        p = define_p(x_vec,y_vec,self.X,self.Y,self.p_cell)
        self.p = p
        mean_sigma =random.uniform(0.09,0.13)
        mean_intensity_art =  random.uniform(self.prop['mean_art'][0],self.prop['mean_art'][1])
        for ind_x, x in enumerate(x_vec[2:-2]):
            ind_x = ind_x+2
            for ind_y, y in enumerate(y_vec[2:-2]):
                ind_y = ind_y+2
                if np.random.choice([0,1],p =[1-p[ind_y,ind_x],p[ind_y,ind_x]]):
                    self.X_art.append((x+np.random.uniform(1,x_w-1))*self.resolution_xy)
                    self.Y_art.append((y+np.random.uniform(1,y_w-1))*self.resolution_xy)
                    self.intensity_art.append(mean_intensity_art  * np.random.normal(1,0.3,1)[0])
                    self.sigma_x_art.append(mean_sigma * np.random.normal(1,0.3,1)[0])
                    self.sigma_y_art.append(2*mean_sigma-self.sigma_x_art[-1])
                    
    def create_box(self,how):
        target = []
        self.difficulties = []
        self.label = []
        if how =='xy':
            for x,y,sigma,i in zip(self.X,self.Y,self.sigma,self.intensity):
                r = np.sqrt(2*sigma*sigma*np.log(1.5*i/self.minimun_intensity))+0.25
                y1_vec = np.max([0,x-r])
                y2_vec = np.min([self.x_vector[-1],x+r])
                x1_vec = np.max([0,y-r])
                x2_vec = np.min([self.y_vector[-1],y+r])
                x1 = self.x_vector.index(min(self.x_vector, key=lambda a:abs(a-x1_vec)))
                x2 = self.x_vector.index(min(self.x_vector, key=lambda a:abs(a-x2_vec)))
                y1 = self.y_vector.index(min(self.y_vector, key=lambda a:abs(a-y1_vec)))
                y2 = self.y_vector.index(min(self.y_vector, key=lambda a:abs(a-y2_vec)))
                target.append([y1/self.img_shape[1],x1/self.img_shape[0],y2/self.img_shape[1],x2/self.img_shape[0]])
                self.difficulties.append(0)
                self.label.append(1)
        if how =='cxcy':
            self.center_raduis = {}
            for x,y,sigma,i in zip(self.X,self.Y,self.sigma,self.intensity):
                r = (np.sqrt(2*sigma*sigma*np.log(1.5*i/self.minimun_intensity))+0.25)/(300*self.resolution_xy)
                x = self.x_vector.index(min(self.x_vector, key=lambda a:abs(a-x)))
           
                y = self.y_vector.index(min(self.y_vector, key=lambda a:abs(a-y)))
                target.append([y/self.img_shape[0],x/self.img_shape[1],r,r])
                self.difficulties.append(0)
                self.label.append(1)
        return target
    def create_circles(self):
        target = []
        self.difficulties = []
        self.label = []
        for x,y,sigma,i in zip(self.X,self.Y,self.sigma,self.intensity):
            r = (np.sqrt(2*sigma*sigma*np.log(1.5*i/self.minimun_intensity)))/self.resolution_xy
            x = self.x_vector.index(min(self.x_vector, key=lambda a:abs(a-x)))
           
            y = self.y_vector.index(min(self.y_vector, key=lambda a:abs(a-y)))
            target.append([y,x,r])
            self.difficulties.append(0)
            self.label.append(1)
            
        return target
    
    
    def create_mask(self,threshold = 0.23):
        mask = np.zeros(self.img_shape,dtype = bool)
        for key,value in self.centrioles.items():
            for _, coor in value.items():
                x_c = coor['X']
                y_c = coor['Y']
                x_closest = self.x_vector.index(min(self.x_vector, key=lambda x:abs(x-x_c)))
                y_closest = self.x_vector.index(min(self.y_vector, key=lambda y:abs(y-y_c)))
                pad = int(threshold*2/0.1025)+1

                for x_ind,x in zip(range(x_closest-pad,x_closest+pad) ,self.x_vector[x_closest-pad:x_closest+pad]):
                    for y_ind,y in zip(range(y_closest-pad,y_closest+pad),self.y_vector[y_closest-pad:y_closest+pad]):
                        if np.linalg.norm([self.x_vector[x_closest]-x,self.y_vector[y_closest]-y])<threshold:
                            mask[x_ind,y_ind] = True         
        return mask
    
    def create_mask_3(self):
        mask = np.zeros((2,)+self.img_shape,dtype = bool)
        for key,value in self.centrioles.items():
            for _, coor in value.items():
                x_c = coor['X']
                y_c = coor['Y']
                z_c = coor['Z']
                x_closest = self.x_vector.index(min(self.x_vector, key=lambda x:abs(x-x_c)))
                y_closest = self.y_vector.index(min(self.y_vector, key=lambda y:abs(y-y_c)))
                z_closest =  self.z_vector.index(min(self.z_vector, key=lambda z:abs(z-z_c)))
                for x_ind,x in zip(range(x_closest-10,x_closest+10) ,self.x_vector[x_closest-10:x_closest+10]):
                    for y_ind,y in zip(range(y_closest-10,y_closest+10),self.y_vector[y_closest-10:y_closest+10]):
                        if np.linalg.norm([x_c-x,y_c-y])<0.15:
                            mask[0,x_ind,y_ind,z_closest] = True
                                       
        for x_c,y_c,z_c in zip(self.X_art,self.Y_art,self.Z_art):
            x_closest = self.x_vector.index(min(self.x_vector, key=lambda x:abs(x-x_c)))
            y_closest = self.x_vector.index(min(self.y_vector, key=lambda y:abs(y-y_c)))
            z_closest =  self.z_vector.index(min(self.z_vector, key=lambda z:abs(z-z_c)))
            for x_ind,x in zip(range(x_closest-10,x_closest+10) ,self.x_vector[x_closest-10:x_closest+10]):
                for y_ind,y in zip(range(y_closest-10,y_closest+10),self.y_vector[y_closest-10:y_closest+10]):
                    if np.linalg.norm([x_c-x,y_c-y])<0.15:
                        mask[1,x_ind,y_ind,z_closest] = True
                            
        return mask

    
    def create_coordinate_1(self,nb_site,threshold):   
        if nb_site == 0:
            return [],[]
        X = [random.uniform(self.padding,self.img_shape[0]*self.resolution_xy-self.padding)]
        Y = [random.uniform(self.padding,self.img_shape[1]*self.resolution_xy-self.padding)]
        for site in range(nb_site-1):
            while True:
                x_new = random.uniform(self.padding,self.img_shape[0]*self.resolution_xy-self.padding)
                y_new = random.uniform(self.padding,self.img_shape[1]*self.resolution_xy-self.padding)
                if self.distance_check(x_new,y_new,X,Y,threshold):
                    X.append(x_new)
                    Y.append(y_new)
                    break
        return X,Y
    
    def gaussian(self,sx,sy,size = 20):
        def gaus2d(x=0, y=0,  sx=1, sy=1):
            return np.exp(-(x**2. / (2. * sx**2.) + y**2. / (2. * sy**2.)))
        x = np.arange(-size*self.resolution_xy,size*self.resolution_xy,self.resolution_xy)
        y =np.arange(-size*self.resolution_xy,size*self.resolution_xy,self.resolution_xy)
        x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
        return ndimage.rotate(gaus2d(x, y, sx, sy),random.randint(0,45),reshape = False)

    def create_coordinate_2(self):
        for site, nb_centriole in self.nb_centrioles.items():
            if nb_centriole >= 2:
                theta = random.uniform(0,2*np.pi)
                r = random.uniform(0.4,1.5) if nb_centriole ==2 else random.uniform(0.8,1.2)
                self.X.append(self.X[site] + r*np.cos(theta))
                self.Y.append(self.Y[site] + r*np.sin(theta))
                self.sigma_x.append(self.law_sigma())
                self.sigma_y.append(2*self.sigma_image -self.sigma_x[-1])
                I_div = random.uniform(1,3) if nb_centriole ==2 else np.random.normal(1,0.2,1)[0]
                self.intensity.append(self.centrioles[site][1]['i']/I_div)
                self.centrioles[site][2] = {'X':self.X[-1],'Y':self.Y[-1],'i':self.intensity[-1],'sigma_x':self.sigma_x[-1],'sigma_y':self.sigma_y[-1],'d':r}
            if nb_centriole == 3:
                theta = random.uniform(theta+np.pi/4,theta+7*np.pi/4)
                r = r* np.random.normal(1,0.1,1)[0]
                self.X.append(self.X[site] + r*np.cos(theta))
                self.Y.append(self.Y[site] + r*np.sin(theta))
                self.sigma_x.append(self.law_sigma())
                self.sigma_y.append(2*self.sigma_image -self.sigma_x[-1])
                I_div = np.random.normal(1,0.1,1)[0]
                self.intensity.append(self.centrioles[site][1]['i']/I_div)
                self.centrioles[site][3] = {'X':self.X[-1],'Y':self.Y[-1],'i':self.intensity[-1],'sigma_x':self.sigma_x[-1],'sigma_y':self.sigma_y[-1],'d':r}
            if nb_centriole == 4:
                r2 = random.uniform(0.4,2)
                theta = theta+np.pi/2 + (r2/r)*random.uniform(-np.pi/4,np.pi/4)
                for i, index in enumerate([-1,site]):
                    r3 = r2 *np.random.normal(1,0.1,1)[0]
                    self.X.append(self.X[index] + r3*np.cos(theta))
                    self.Y.append(self.Y[index] + r3*np.sin(theta))
                    self.sigma_x.append(self.law_sigma())
                    self.sigma_y.append(2*self.sigma_image -self.sigma_x[index])
                    I_div = np.random.normal(1,0.1,1)[0]
                    self.intensity.append(self.centrioles[site][1]['i']/I_div)
                    self.centrioles[site][3+i] = {'X':self.X[-1],'Y':self.Y[-1],'i':self.intensity[-1],'sigma_x':self.sigma_x[-1],'sigma_y':self.sigma_y[-1],'d':r2}
                    
                    
    def add_gaussian_art(self):
        img_gaussian = np.zeros(self.img.shape,dtype = 'float32')
        window = 4
        for x_c,y_c,intensity_c,sigma_x,sigma_y in zip(self.X_art,self.Y_art,self.intensity_art,self.sigma_x_art,self.sigma_y_art):
            x_closest = self.x_vector.index(min(self.x_vector, key=lambda x:abs(x-x_c)))
            y_closest = self.y_vector.index(min(self.y_vector, key=lambda y:abs(y-y_c)))
            img_gaussian[x_closest-window:x_closest+window,y_closest-window:y_closest+window] =img_gaussian[x_closest-window:x_closest+window,y_closest-window:y_closest+window]+ intensity_c*self.gaussian(sx = sigma_x,sy = sigma_y,size = window)
        return img_gaussian
        
    
    def add_gaussian(self):
        img_gaussian = np.zeros(self.img.shape,dtype = 'float32')
        window = 20
        for x_c,y_c,intensity_c,sigma_x,sigma_y in zip(self.X,self.Y,self.intensity,self.sigma_x,self.sigma_y):
            x_closest = self.x_vector.index(min(self.x_vector, key=lambda x:abs(x-x_c)))
            y_closest = self.y_vector.index(min(self.y_vector, key=lambda y:abs(y-y_c)))
            img_gaussian[x_closest-window:x_closest+window,y_closest-window:y_closest+window] =img_gaussian[x_closest-window:x_closest+window,y_closest-window:y_closest+window]+ intensity_c*self.gaussian(sx = sigma_x,sy = sigma_y,size = window)
        return img_gaussian

    
    def get_data(self):
        img = (self.img-self.mean)/ 2**16
        img[img<0] = 0
        img[img>1] = 1
        return  img