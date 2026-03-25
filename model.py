import numpy as np
import torch 
from torch import nn
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchmetrics


class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, num_classes=num_classes, kernel_size=3)
        self.relu1 = nn.ReLU()
        self.avg_pool = nn.AvgPool2d()
        self.conv2 = nn.Conv2d(in_channels, num_classes, )
        self.fc1 = nn.Linear()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.avg_pool(x)
        x = self.conv2(x)
        x = x.view()
        x = self.fc1(x)

        return x
    
