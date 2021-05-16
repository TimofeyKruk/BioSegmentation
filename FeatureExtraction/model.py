import os
import numpy as np
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet


class FEModel(nn.Module, ):
    def __init__(self, model_name="efficientnet-b0"):
        super().__init__()
        self.model_name = model_name

        self.model = EfficientNet.from_name(model_name)
        self.model._fc = nn.Linear(1280, out_features=256)

        self.out_layer = nn.Linear(in_features=256, out_features=2)

    def forward(self, x):
        x = self.model(x)
        return self.out_layer(x)

    def get_features(self,x):
        return self.model(x)


if __name__ == '__main__':
    model = EfficientNet.from_name("efficientnet-b0")
    print(model)
