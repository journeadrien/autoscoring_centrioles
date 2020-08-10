# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 12:31:36 2020

@author: journe
"""
from model import get_model
import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.io import imread
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset , DataLoader
from engine import validate, train
import torch
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-j', '--num-workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=3, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    help='batch-size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    help='initial learning rate', dest='lr')
parser.add_argument('-p', '--print-freq', default=2, type=int,
                    help='print frequency (default: 10)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none),\
                    (keyword : "last" take the last checkpoint)')
parser.add_argument('--gpu', dest='gpu', action='store_true',
                    help='Train on gpu if given')
parser.add_argument('--cw', dest='class_weigth', action='store_true',
                    help='Train on gpu if given')
parser.add_argument('-model', default='efficientnet', type=str, 
                    help='name of the model')
parser.add_argument('-s','--step-factor', default=10, type=int, 
                    help='step at which lr decrease')
def load_annotation():
    cycle_phase_folder = "E:\\Adrien\\data\\dataset\\cycle_phase" 
    lst_dir = os.listdir(cycle_phase_folder)
    files = [x.split('.')[0] for x in lst_dir if x.endswith('.png')]
    files.sort()
    classes = []
    for file in files:
        with open(os.path.join(cycle_phase_folder,file+'.txt')) as f:
            d = f.read()
            classes.append(int(d[0]))
    return files, classes

def split_train_val(files, classes, threshold):
    train_files, val_files, train_classes, val_classes = train_test_split(files, classes, test_size=threshold,
                                               random_state=42, shuffle=True)
    return train_files, val_files, train_classes, val_classes



class MyDataset(Dataset):
    def __init__(self, files, classes, class_weight, train = True):
        self.cycle_phase_folder = "E:\\Adrien\\data\\dataset\\cycle_phase" 
        self.files = files
        self.classes = classes
        self.class_weight = class_weight
        nomalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        if train:
            self.transf = transforms.Compose([
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(45),
            transforms.ToTensor(),
            nomalize
        ])
        else:
            self.transf = transforms.Compose([transforms.ToTensor(),
                                              nomalize])

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # image = imread(os.path.join(self.cycle_phase_folder, self.files[idx]+'.tif'), plugin='tifffile')
        image = Image.open(os.path.join(self.cycle_phase_folder, self.files[idx]+'.png'))
        # image = Image.fromarray(image, mode='I;16')
        return self.transf(image), self.classes[idx], self.class_weight[self.classes[idx]]
    
def get_class_weigth(classes, args):
    nb_samples = len(classes)
    nb_classes = np.unique(classes)
    if args.class_weigth:
        classes_array = np.array(classes)
        class_weigth = {class_ : (classes_array == class_).sum() / nb_samples for class_ in nb_classes}
        max_ = np.max(list(class_weigth.values()))
        class_weigth = {class_ : max_/value for class_, value in class_weigth.items()}
        return class_weigth
    return {class_ : 1 for class_ in nb_classes}
        
        
def get_device(args):

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.gpu:
        torch.cuda.empty_cache()
        torch.autograd.set_detect_anomaly(True)
        print("device : " + str(args.device))

        if not torch.cuda.is_available():
            args.gpu = False
            print(10 * "*" + 'GPU not working !!!' + 10 * "*")
    else:
        print('Use CPU')

    return args.device

def main(args):
    start = time.time()
    args.nb_classes = 4
    files, classes = load_annotation()
    class_weigth = get_class_weigth(classes, args)
    train_files, val_files, train_classes, val_classes = split_train_val(files, classes, 0.15)
    model = get_model(args)
    args.device = get_device(args)
    
    model.to(args.device)
    train_dataset = MyDataset(train_files, train_classes, class_weigth)
    val_dataset = MyDataset(val_files, val_classes, class_weigth, train = False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                 num_workers=args.num_workers, shuffle=True,
                 pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                 num_workers=args.num_workers,
                 pin_memory=True)
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=0.9, weight_decay=0.0005)
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=args.step_factor,
                                                   gamma=0.4)
    
    # let's train it for 10 epochs
    # model.load_state_dict(torch.load('model/64.pt'))
    
    for epoch in range(1, args.epochs):
        
        # train for one epoch, printing every 10 iterations
        train(train_loader, model, criterion, optimizer, epoch, args)
        # update the learning rate
        _ = validate(val_loader, model, criterion, args)
        lr_scheduler.step()

        torch.save(model.state_dict(), 'model/0.pt')

    print('Elapsed time', round((time.time() - start)/60, 1), 'minutes')
    
    
if __name__ == '__main__':
    #  python train.py --num-workers 8  --gpu --batch-size 16 --epochs 50 --lr 0.005 --cw --step-factor 5
    args = parser.parse_args()
    main(args)