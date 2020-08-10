# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:23:04 2020

@author: journe
"""


import cv2
import numpy as np
import os
import numpy as np
import torch
from augment_preprocess import fix_crop_transform, random_crop_transform, relabel_multi_mask
from augment_preprocess import random_shift_scale_rotate_transform, clean_masks
from config import Config, load_img
import h5py
import utils
from torchvision import transforms
import matplotlib.pyplot as plt
normilize  =transforms.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]) 


class Dataset(torch.utils.data.Dataset):
    """The base class for dataset classes.
    To use it, create a new class that adds functions specific to the dataset
    you want to use. For example:
    class CatsAndDogsDataset(Dataset):
        def load_cats_and_dogs(self):
            ...
        def load_mask(self, image_id):
            ...
        def image_reference(self, image_id):
            ...
    See COCODataset and ShapesDataset as examples.
    """

    def __init__(self, class_map=None):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {
            "id": image_id,
            "source": source,
            "path": path,
        }
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def image_reference(self, image_id):
        """Return a link to the image in its source Website or details about
        the image that help looking it up or debugging it.
        Override for your dataset, but pass to this function
        if you encounter images not in your dataset.
        """
        return ""

    def prepare(self, class_map=None):
        """Prepares the Dataset class for use.
        TODO: class map is not supported yet. When done, it should handle mapping
              classes from different datasets to the same class ID.
        """

        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)
        self.class_ids = np.arange(self.num_classes)
        self.class_names = [clean_name(c["name"]) for c in self.class_info]
        self.num_images = len(self.image_info)
        self._image_ids = np.arange(self.num_images)

        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}

        # Map sources to class_ids they support
        self.sources = list(set([i['source'] for i in self.class_info]))
        self.source_class_ids = {}
        # Loop over datasets
        for source in self.sources:
            self.source_class_ids[source] = []
            # Find classes that belong to this dataset
            for i, info in enumerate(self.class_info):
                # Include BG class in all datasets
                if i == 0 or source == info['source']:
                    self.source_class_ids[source].append(i)

    def map_source_class_id(self, source_class_id):
        """Takes a source class ID and returns the int class ID assigned to it.
        For example:
        dataset.map_source_class_id("coco.12") -> 23
        """
        return self.class_from_source_map[source_class_id]

    def get_source_class_id(self, class_id, source):
        """Map an internal class ID to the corresponding class ID in the source dataset."""
        info = self.class_info[class_id]
        assert info['source'] == source
        return info['id']

    def append_data(self, class_info, image_info):
        self.external_to_class_id = {}
        for i, c in enumerate(self.class_info):
            for ds, id in c["map"]:
                self.external_to_class_id[ds + str(id)] = i

        # Map external image IDs to internal ones.
        self.external_to_image_id = {}
        for i, info in enumerate(self.image_info):
            self.external_to_image_id[info["ds"] + str(info["id"])] = i

    @property
    def image_ids(self):
        return self._image_ids

    def source_image_link(self, image_id):
        """Returns the path or URL to the image.
        Override this to return a URL to the image if it's availble online for easy
        debugging.
        """
        return self.image_info[image_id]["path"]


class NucleiDataset(Dataset):

    def load_shapes(self, id_list, train_path,mode):
        """initialize the class with dataset info.
        """
        self.transform = transforms.Compose([
                            transforms.ToTensor()])
                           
        self.config = Config()
        if mode == 'train':
            self.augment = True
        else:
            self.augment = False
        # Add classes
        self.add_class('images', 1, "nucleus")
        self.train_path = train_path
        self.imgs = id_list
        # Add images
        for i, id_ in enumerate(id_list):
            self.add_image('images', image_id=i, path=None,
                           img_name=id_)
            
            
    def load_image(self, image_id, color):
        """Load image from directory
        """

        info = self.image_info[image_id]
        path = self.train_path + info['img_name'] + \
            '/images/' + info['img_name'] + '.png'

        img = load_img(path, color=color)

        return img

    def image_reference(self, image_id):
        """Return the images data of the image."""
        info = self.image_info[image_id]
        if info["source"] == 'images':
            return info['images']
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for images of the given image ID.
        """

        info = self.image_info[image_id]

        path = self.train_path + info['img_name'] + \
            '/masks/' + info['img_name'] + '.h5'
        if os.path.exists(path):
            # For faster data loading run augment_preprocess.py file first
            # That should save masks in a single h5 file
            with h5py.File(path, "r") as hf:
                mask = hf["arr"][()]
        else:
            path = self.train_path + info['img_name']
            mask = []
            for mask_file in next(os.walk(path + '/masks/'))[2]:
                if 'png' in mask_file:
                    mask_ = cv2.imread(path + '/masks/' + mask_file, 0)
                    mask_ = np.where(mask_ > 128, 1, 0)
                    # Fill holes in the mask
                    mask_ = binary_fill_holes(mask_).astype(np.int32)
                    # Add mask only if its area is larger than one pixel
                    if np.sum(mask_) >= 1:
                        mask.append(np.squeeze(mask_))

            mask = np.stack(mask, axis=-1)
            mask = mask.astype(np.uint8)

        # Class ids: all ones since all are foreground objects
        class_ids = np.ones(mask.shape[2])

        return mask.astype(np.uint8), class_ids.astype(np.int8)

    def __getitem__(self, image_id):
        # load images ad masks
        # Load image and mask
        image = self.load_image(image_id, self.config.IMAGE_COLOR)
        masks, class_ids = self.load_mask(image_id)
        temp_image, temp_mask = random_crop_transform(image, masks, 512, 512)
        keep_ind = np.where(np.sum(temp_mask, axis=(0, 1)) > 0)[0]
        if len(keep_ind) > 0:
            image = temp_image.copy()
            masks = temp_mask[:,:,keep_ind]

        

        # Random scaling
        if self.augment and self.config.SCALE and np.random.rand() < 0.9:
            H, W = masks.shape[:2]
             #multi_mask = np.sum(mask*np.arange(1, mask.shape[-1]+1), axis=-1)
            temp_image, temp_mask = random_shift_scale_rotate_transform(image.copy(), masks.copy(),
                                                                         shift_limit=[-0.0625, 0.0625], scale_limit=[1/1.2, 1.2],
                                                                         rotate_limit=[-15, 15])
            temp_mask = temp_mask if len(temp_mask.shape) ==3 else np.expand_dims(temp_mask,axis = 2)
             #temp_mask = np.repeat(multi_mask[:, :, np.newaxis], multi_mask.max(), axis=-1)
             #temp_mask = np.equal(temp_mask, np.ones_like(temp_mask)*np.arange(1, multi_mask.max()+1)).astype(np.uint8)

            keep_ind = np.where(np.sum(temp_mask, axis=(0, 1)) > 0)[0]
            if len(keep_ind) > 0:
                image = temp_image
                masks = temp_mask[:,:,keep_ind]

         #print(masks.shape)
        # Random cropping
        if (masks.shape[2] > 70) or (self.augment and self.config.CROP and np.random.rand() < 0.3):

            height = self.config.CROP_SHAPE[1]
            width = self.config.CROP_SHAPE[0]

            temp_image, temp_mask = random_crop_transform(image, masks, height, width)
            keep_ind = np.where(np.sum(temp_mask, axis=(0, 1)) > 0)[0]
            if len(keep_ind) > 0:
                image = temp_image.copy()
                masks = temp_mask[:,:,keep_ind]

        shape = image.shape
        image, window, scale, padding = utils.resize_image(
            image,
            min_dim=self.config.IMAGE_MIN_DIM,
            max_dim=self.config.IMAGE_MAX_DIM,
            padding=self.config.IMAGE_PADDING)
        masks = utils.resize_mask(masks, scale, padding)
        
    
        
        # Random horizontal and vertical flips.
        if self.augment:

            # horizontal
            if np.random.rand() < 0.5:
                image = np.fliplr(image)
                masks = np.fliplr(masks)
            # Vertical
            if np.random.rand() < 0.5:
                image = np.flipud(image)
                masks = np.flipud(masks)

            # Random 90 degree rotation
            if np.random.rand() < 0.5:
                angle = np.random.choice([1, -1])
                image = np.rot90(image, k=angle, axes=(0, 1))
                masks = np.rot90(masks, k=angle, axes=(0, 1))

            # # Random Gaussian blur
            # if random.randint(0, 1):
            #     sigma = np.random.uniform(1.5,2.5)
            #     image = cv2.GaussianBlur(image, (33, 33), sigma)
        
        
    
        keep_ind = np.where(np.sum(masks, axis=(0, 1)) > 0)[0]
        if len(keep_ind) > 0:
            masks = masks[:,:,keep_ind]
         #image =  np.swapaxes(np.swapaxes(image,1,2),0,1)
        
        masks = masks.transpose((2, 0, 1))
         # get bounding box coordinates for each mask
        num_objs = masks.shape[0]
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1]-1)
            xmax = np.max(pos[1])
            ymin = np.min(pos[0]-1)
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
            
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image  = self.transform((image/image.max()).astype(np.float32))
        image_id = torch.tensor([image_id])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
       
        return image, target

    def __len__(self):
        return len(self.imgs)


class NucleiExpDataset(NucleiDataset):

    def load_shapes(self, ir,transform = True):
        """initialize the class with dataset info.
        """
        self.config = Config()
        self.ir = ir

        if transform:
            self.transform = transforms.Compose([
                                transforms.ToTensor()])
        else:
            self.transform = None
      
        self.add_class('images', 1, "nucleus")
        self.imgs = list(ir.pos.keys())
        # Add images
        for i,pos in enumerate(ir.pos.keys()):
            self.add_image('images', image_id=i, path=None,
                           img_name=pos)
            
            
    def load_image(self, image_id):
        """Load image from directory
        """

        info = self.image_info[image_id]
        pos = info['img_name']
        img = self.ir.get_image(pos,'DAPI')
        img = np.max(img,0)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # img = np.array([np.max(images,0),np.mean(images,0),np.std(images,0)])
        return img


    def __getitem__(self, image_id):
        image = self.load_image(image_id)
        image, window, scale, padding = utils.resize_image(
            image,
            min_dim=self.config.IMAGE_MIN_DIM,
            max_dim=self.config.IMAGE_MAX_DIM,
            padding=False)
        image = (image/image.max()).astype(np.float32)
        image  = self.transform(image) if self.transform is not None else image
        return image