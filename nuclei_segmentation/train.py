# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:20:52 2020

@author: journe
"""
from dataset import NucleiDataset
import os
from engine import train_one_epoch, evaluate
from model import get_model_instance_segmentation
import utils
from config import DATASET_PATH
import os.path as op
import torch
import time

import torchvision





def train_validation_split(train_path,classes_csv , seed=10, test_size=0.1):

    """
    Split the dataset into train and validation sets.
    External data and mosaics are directly appended to training set.
    """
    from sklearn.model_selection import train_test_split
    import pandas as pd 
    
    image_ids = list(
        filter(lambda x: ('mosaic' not in x) and ('TCGA' not in x), os.listdir(train_path)))
    mosaic_ids = list(filter(lambda x: 'mosaic' in x, os.listdir(train_path)))
    external_ids = list(filter(lambda x: 'TCGA' in x, os.listdir(train_path)))

    # Load and preprocess the dataset with train image modalities
    df = pd.read_csv(classes_csv)
    df['labels'] = df['foreground'].astype(str) + df['background']
    df['filename'] = df['filename'].apply(lambda x: x[:-4])
    df = df.set_index('filename')
    df = df.loc[image_ids]
    df = df.loc[df.background == 'black']
    # Split training set based on provided image modalities
    # This ensures that model validates on all image modalities.
    train_list, val_list = train_test_split(df.index, test_size=test_size,
                                            random_state=seed, stratify=df['labels'])

    # Add external data and mos ids to training list
    train_list = list(train_list)
    val_list = list(val_list)

    return list(df.index), val_list

if __name__ == '__main__':


    start = time.time()

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # our dataset has two classes only - background and person
    num_classes = 2
    
    
    
    train_path = op.join(DATASET_PATH,'stage1_train/')
    classes_csv = op.join(DATASET_PATH,'classes.csv')

    
    # Split the training set into training and validation
    train_list, val_list = train_validation_split(
        train_path,classes_csv , seed=11, test_size=0.01)
    
    # initialize training dataset
    dataset_train = NucleiDataset()
    dataset_train.load_shapes(train_list, train_path,'train')
    dataset_train.prepare()
    
    # initialize validation dataset
    dataset_val = NucleiDataset()
    dataset_val.load_shapes(val_list, train_path,'val')
    dataset_val.prepare()
    
    
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=1, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)
    
    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)
    
    # move model to the right device
    model.to(device)
    
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=5,
                                                   gamma=0.2)
    
    # let's train it for 10 epochs
    num_epochs = 50
    
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        # evaluate(model, data_loader_test, device=device)

        torch.save(model.state_dict(), 'model/0.pt')

    print('Elapsed time', round((time.time() - start)/60, 1), 'minutes')
