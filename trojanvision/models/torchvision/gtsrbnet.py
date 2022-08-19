#!/usr/bin/env python3
import torch

from trojanvision.models.imagemodel import _ImageModel, ImageModel

import torch.nn as nn
from collections import OrderedDict


class _GTSRBNet(_ImageModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(3, 32, 3)),
            ('relu1', nn.ReLU(True)),
            ('conv2', nn.Conv2d(32, 32, 3)),
            ('relu2', nn.ReLU(True)),
            ('pool1', nn.MaxPool2d(2, 2)),
            ('conv3', nn.Conv2d(32, 64, 3)),
            ('relu3', nn.ReLU(True)),
            ('conv4', nn.Conv2d(64, 64, 3)),
            ('relu4', nn.ReLU(True)),
            ('pool2', nn.MaxPool2d(2, 2)),
            ('conv5', nn.Conv2d(64, 128, 3)),
            ('relu5', nn.ReLU(True)),
            ('conv6', nn.Conv2d(128, 128, 3)),
            ('relu6', nn.ReLU(True)),
            ('pool3', nn.MaxPool2d(2, 2)),
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(32 * 4 * 4, 512)),
            ('relu1', nn.ReLU(True)),
            ('fc2', nn.Linear(512, 43)),
            ('softmax', nn.Softmax(dim=1 )),

        ]))
        self.name = 'mnistnet'
        if kwargs.get('quant'):
            if kwargs.get('ptsq'):
                self.quant = torch.quantization.QuantStub()
                self.dequant = torch.quantization.DeQuantStub()


class GTSRBNet(ImageModel):
    available_models = ['mnistnet']

    def __init__(self, name: str = 'net', model: type[_GTSRBNet] = _GTSRBNet, **kwargs):
        super().__init__(name=name, model=model, **kwargs)
