import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchsummary import summary
from tqdm import tqdm
from utils import *


class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
			nn.Conv2d(1, 64, 5, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Dropout(0.25),           
			
            nn.MaxPool2d(2, 2)

			)
			
        self.conv2 = nn.Sequential(

			nn.Conv2d(64, 32, 1, padding=0),

            nn.Conv2d(32, 32, 3, padding=0),
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 10, 1, padding=0),

			nn.Conv2d(10, 10, 5),
            nn.ReLU(),
            nn.BatchNorm2d(10),

            nn.MaxPool2d(2, 2)

			)
        
        self.fc = nn.Sequential(
            nn.Linear(90, 10)
			)

		
    def forward(self, x):                                 													# Channel Size | RF | Jump Parameter
        x = self.conv1(x)                      																# 28 > 26 | 1>3 | 1>1
        x = self.conv2(x)    							   													# 26 > 24 > 12 | 3>5>6 | 1>1>2		
        x = x.view(-1, 90) 
        x = self.fc(x)
        x = F.log_softmax(x, dim=1)               													
        
        return x


def draw_graphs(train_losses, train_acc, test_losses, test_acc):
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")