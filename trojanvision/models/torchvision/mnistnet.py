#!/usr/bin/env python3
import torch

from trojanvision.models.imagemodel import _ImageModel, ImageModel

import torch.nn as nn
from collections import OrderedDict


class _MnistNet(_ImageModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 16, 5)),
            ('relu1', nn.ReLU(True)),
            ('pool1', nn.MaxPool2d(2, 2)),
            ('conv2', nn.Conv2d(16, 32, 5)),
            ('relu2', nn.ReLU(True)),
            ('pool2', nn.MaxPool2d(2, 2)),
        ]))
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(32 * 4 * 4, 512)),
            ('relu1', nn.ReLU(True)),
            ('fc2', nn.Linear(512, 10)),
            ('softmax', nn.Softmax(dim=1)),
        ]))
        self.name = 'mnistnet'
        if kwargs.get('quant'):
            if kwargs.get('ptsq'):
                self.quant = torch.quantization.QuantStub()
                self.dequant = torch.quantization.DeQuantStub()


class MnistNet(ImageModel):
    available_models = ['mnistnet']

    def __init__(self, name: str = 'net', model: type[_MnistNet] = _MnistNet, **kwargs):
        super().__init__(name=name, model=model, **kwargs)
