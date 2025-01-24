import torch
import torch.nn as nn
from utils.Config import Config
from utils.weight_init import weight_init_kaiming
from torchvision import models
import os
import numpy as np

class ResNet(nn.Module):
    def __init__(self, pre_trained=True, n_class=200, model_choice=50):
        super(ResNet, self).__init__()
        self.n_class = n_class
        self.base_model = self._model_choice(pre_trained, model_choice)
        self.base_model.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.base_model.fc = nn.Linear(512*Config.expansion, n_class)
        self.base_model.fc.apply(weight_init_kaiming)

    def forward(self, x):
        N = x.size(0)
        assert x.size() == (N, 3, 448, 448)
        x = self.base_model(x)
        assert x.size() == (N, self.n_class)
        return x

    def _model_choice(self, pre_trained, model_choice):
        if model_choice == 50:
            return models.resnet50(pretrained=pre_trained)
        elif model_choice == 101:
            return models.resnet101(pretrained=pre_trained)
