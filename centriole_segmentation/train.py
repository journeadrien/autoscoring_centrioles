# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 11:20:52 2020

@author: journe
"""
from dataset import get_train_valid_loader
import os
from engine import train_one_epoch, evaluate2, evaluate
import utils
import os.path as op
import torch
from model import get_model_mobile_net,get_model_resnet
import time

import torchvision







if __name__ == '__main__':
    size = 10
    channel = 'GFP' if size == 8 else 'RFP' if  size == 6 else 'Cy5'
    print(channel)
    start = time.time()
    model = get_model_resnet(last_channel = 64,sb=(int(size/2)), box_score_thresh = 0.8,box_nms_thresh = 0.35)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    train_loader, val_loader = get_train_valid_loader(epoch_size = 600, batch_size=4,channel = channel,mode = 'annotate',sb=size,
                                              split=True, shuffle=True,
                                              num_workers=4,
                                              val_ratio=0.1)
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=4,
                                                   gamma=0.4)
    
    # let's train it for 10 epochs
    num_epochs = 25
    # model.load_state_dict(torch.load('model/64.pt'))
    for epoch in range(num_epochs):
        
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=20)
        # update the learning rate
        evaluate(model,val_loader,device)
        evaluate2(model,val_loader,device)
        lr_scheduler.step()

        torch.save(model.state_dict(), 'model/resnet_'+str(size)+'.pt')

    print('Elapsed time', round((time.time() - start)/60, 1), 'minutes')
