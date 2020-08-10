# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 12:38:59 2020

@author: journe
"""

from efficientnet_pytorch import EfficientNet
import torchvision.models as models
import torch.nn as nn
from collections import OrderedDict

class enetv2(nn.Module):
    def __init__(self, backbone, out_dim):
        super(enetv2, self).__init__()
        self.enet = EfficientNet.from_pretrained(backbone)

        self.myfc = nn.Linear(self.enet._fc.in_features, out_dim)
        self.enet._fc = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        x = self.myfc(x)
        return x
    
def get_efficientnet(out_dim):
    model = EfficientNet.from_pretrained('efficientnet-b3')
    in_features = model._fc.in_features
    model._fc = nn.Sequential(OrderedDict([
        ('fc1' , nn.Linear(in_features, 512)),
        ('relu1' , nn.ReLU()),
        ('fc2' , nn.Linear(512, 100)),
        ('relu2' , nn.ReLU()),
         ('fc3' , nn.Linear(100, out_dim))
        ]))
    return model

def get_resnet(out_dim):
    model = models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    model._fc = nn.Sequential(OrderedDict([
        ('fc' , nn.Linear(in_features, 100)),
        ('relu1', nn.ReLU()),
         ('fc2' , nn.Linear(100, out_dim))
        ]))
    return model

def get_model(args):
    if args.model == 'efficientnet':
        return get_efficientnet(args.nb_classes)

    return get_resnet(args.nb_classes)