
import torch
import torch.nn as nn
from torchvision import models

from .alexnet_micro import alexnet 
from .resnet_micro import resnet18
from .resnet_micro import resnet34 
from .densenet_micro import densenet121 

def load_model(name, outputsize, pretrained=None):

    if pretrained:
        pretrained = True
    else:
        pretrained = False

    if name.lower() in 'alexnet':
        model = alexnet(pretrained=pretrained)
        model.classifier = nn.Linear(256, outputsize)
    elif name.lower() in 'resnet18':
        model = resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, outputsize)
    elif name.lower() in 'resnet34':
        model = resnet34(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, outputsize)
    elif name.lower() in 'densenet121':
        model = densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, outputsize)

    return model



