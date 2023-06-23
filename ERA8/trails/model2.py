import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchsummary import summary
from tqdm import tqdm
from utils import *

GROUP_SIZE = 2

class Net(nn.Module):
    def __init__(self, norm):
        super(Net, self).__init__()


        # Input Block
        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
        ) # output_size = 32

        out_channels=32
        if norm == 'bn':
            self.n1 = nn.BatchNorm2d(out_channels)
        elif norm == 'gn':
            self.n1 = nn.GroupNorm(GROUP_SIZE, out_channels)
        elif norm == 'ln':
            self.n1 = nn.GroupNorm(1, out_channels)

        self.convblock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
        ) # output_size = 32

        out_channels = 64
        if norm == 'bn':
            self.n2 = nn.BatchNorm2d(out_channels)
        elif norm == 'gn':
            self.n2 = nn.GroupNorm(GROUP_SIZE, out_channels)
        elif norm == 'ln':
            self.n2 = nn.GroupNorm(1, out_channels)



        # TRANSITION BLOCK 1
        self.convblock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=16, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
        ) # output_size = 32

        out_channels = 16
        if norm == 'bn':
            self.n3 = nn.BatchNorm2d(out_channels)
        elif norm == 'gn':
            self.n3 = nn.GroupNorm(GROUP_SIZE, out_channels)
        elif norm == 'ln':
            self.n3 = nn.GroupNorm(1, out_channels)

        self.pool1 = nn.MaxPool2d(2, 2) # output_size = 16



        # CONVOLUTION BLOCK 2
        self.convblock4 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
        ) # output_size = 16

        out_channels=32
        if norm == 'bn':
            self.n4 = nn.BatchNorm2d(out_channels)
        elif norm == 'gn':
            self.n4 = nn.GroupNorm(GROUP_SIZE, out_channels)
        elif norm == 'ln':
            self.n4 = nn.GroupNorm(1, out_channels)

        self.convblock5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
        ) # output_size = 16

        out_channels=32
        if norm == 'bn':
            self.n5 = nn.BatchNorm2d(out_channels)
        elif norm == 'gn':
            self.n5 = nn.GroupNorm(GROUP_SIZE, out_channels)
        elif norm == 'ln':
            self.n5 = nn.GroupNorm(1, out_channels)

        self.convblock6 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # output_size = 14

        out_channels=32
        if norm == 'bn':
            self.n6 = nn.BatchNorm2d(out_channels)
        elif norm == 'gn':
            self.n6 = nn.GroupNorm(GROUP_SIZE, out_channels)
        elif norm == 'ln':
            self.n6 = nn.GroupNorm(1, out_channels)



        # TRANSITION BLOCK 2
        self.convblock7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=10, kernel_size=(1, 1), padding=0, bias=False),
            nn.ReLU(),
        ) # output_size = 14

        out_channels=10
        if norm == 'bn':
            self.n7 = nn.BatchNorm2d(out_channels)
        elif norm == 'gn':
            self.n7 = nn.GroupNorm(GROUP_SIZE, out_channels)
        elif norm == 'ln':
            self.n7 = nn.GroupNorm(1, out_channels)

        self.pool2 = nn.MaxPool2d(2, 2) # output_size = 7



       # CONVOLUTION BLOCK 3
        self.convblock8 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=1, bias=False),
            nn.ReLU(),
        ) # output_size = 7

        out_channels=10
        if norm == 'bn':
            self.n8 = nn.BatchNorm2d(out_channels)
        elif norm == 'gn':
            self.n8 = nn.GroupNorm(GROUP_SIZE, out_channels)
        elif norm == 'ln':
            self.n8 = nn.GroupNorm(1, out_channels)

        self.convblock9 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # output_size = 5

        out_channels=10
        if norm == 'bn':
            self.n9 = nn.BatchNorm2d(out_channels)
        elif norm == 'gn':
            self.n9 = nn.GroupNorm(GROUP_SIZE, out_channels)
        elif norm == 'ln':
            self.n9 = nn.GroupNorm(1, out_channels)

        self.convblock10 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), padding=0, bias=False),
            nn.ReLU(),
        ) # output_size = 3

        out_channels=10
        if norm == 'bn':
            self.n10 = nn.BatchNorm2d(out_channels)
        elif norm == 'gn':
            self.n10 = nn.GroupNorm(GROUP_SIZE, out_channels)
        elif norm == 'ln':
            self.n10 = nn.GroupNorm(1, out_channels)



        # OUTPUT BLOCK

        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=3)) # output_size = 1

        self.convblock11 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(1, 1), padding=0, bias=False)
        )
        self.dropout = nn.Dropout(0.1)
   

    def forward(self, x):
        x = self.convblock1(x)
        x = self.n1(x)
        x = self.dropout(x)
        x = self.convblock2(x)
        x = self.n2(x)
        x = self.dropout(x)

        x = self.convblock3(x)
        x = self.n3(x)
        x = self.pool1(x)

        x = self.convblock4(x)
        x = self.n4(x)
        x = self.dropout(x)
        x = self.convblock5(x)
        x = self.n5(x)
        x = self.dropout(x)
        x = self.convblock6(x)
        x = self.n6(x)
        x = self.dropout(x)

        x = self.convblock7(x)
        x = self.n7(x)
        x = self.pool1(x)

        x = self.convblock8(x)
        x = self.n8(x)
        x = self.dropout(x)
        x = self.convblock9(x)
        x = self.n9(x)
        x = self.dropout(x)
        x = self.convblock10(x)
        x = self.n10(x)

        x = self.gap(x)
        #x = self.convblock11(x)
        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)