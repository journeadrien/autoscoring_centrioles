# -*- coding: utf-8 -*-
"""
Created on Sat May  2 18:39:48 2020

@author: journe
"""
import cv2
from glob import glob
import numpy as np


def saturate_img(img,factor = 1000):
    img = img/img.max()
    img = img*factor
    img[img<0] = 0
    img[img>1] = 1
    return img

def DAPI_image(images,how = 'maximun projection'):
    if how == 'maximum projection':
        return np.max(images, axis=0)
    elif how == 'focus':
        return images[int(images.shape[0]/2)]
    
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


def resize_image(image, min_dim=None, max_dim=None, padding=False):
    """
    Resizes an image keeping the aspect ratio.
    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    padding: If true, pads image with zeros so it's size is max_dim x max_dim
    Returns:
    image: the resized image
    window: (y1, x1, y2, x2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1

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
    # Need padding?
    if padding:
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return image, window, scale, padding