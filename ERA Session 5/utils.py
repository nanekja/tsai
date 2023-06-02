import sys
import os
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

def get_device():
  '''
  This method returns the device in use.
  If cuda(gpu) is available it would return that, otherwise it would return cpu.
  '''
  use_cuda = torch.cuda.is_available()
  # We are checking with PyTorch whether cuda library is available. This cuda library will help to accelerate execution of the model on the GPU
  print(torch.device("cuda" if use_cuda else "cpu"))
  

def train_transforms():
  # Train data transformations
  train_transforms = transforms.Compose([
    # This is the Transform (T) part of the ETL process where we are transforming the data into tensors
    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
    # It does a crop in the center in a random fashion
    transforms.Resize((28, 28)),
    # resizes the image to 28 x 28
    transforms.RandomRotation((-15., 15.), fill=0),
    # performs a rotation in a random way based on parameters entered
    transforms.ToTensor(),
    # Immediately after downloading the dataset, we are converting it into tensor
    # Converting to Tensor facilitates following: a) It allows porting data to GPU b) It standardizes the values from 0-255 to 0-1
    transforms.Normalize((0.1307,), (0.3081,)),
    # Normalization is done to bring lot of bright or dull images to similar level of brightness of other images
    ])
  return train_transforms

def test_transforms():
  test_transforms = transforms.Compose([
    # This is the Transform (T) part of the ETL process where we are transforming the data into tensors
    transforms.ToTensor(),
    # Immediately after downloading the dataset, we are converting it into tensor
    # Converting to Tensor facilitates following: a) It allows porting data to GPU b) It standardizes the values from 0-255 to 0-1
    transforms.Normalize((0.1307,), (0.3081,))
    # Normalization is done to bring lot of bright or dull images to similar level of brightness of other images
    ])
  return test_transforms


def data_loaders(batch_size):
  train_data = datasets.MNIST('../data', train=True, download=True, transform=train_transforms)
  # The dataset named "MNIST" is being called from within the torchvision available datasets
  # MNIST is a dataset containing handwritten digits
  # The training part of the dataset is downloaded to ../data location
  # This is also the Extract (E) part of the ETL process where we are going to extract the dataset from raw data
  test_data = datasets.MNIST('../data', train=False, download=True, transform=test_transforms)
  # The test part of the dataset is downloaded to ../data location
  
  kwargs = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 2, 'pin_memory': True}

  train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
  # trainloader is sort of "for" loop for us which will allow to look at or load a lot of images (~batch_size) at same time
  # torch.utils.data.DataLoader wraps a dataset and provides access to the underlying data
  test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
  # test till help to check accuracy of our model
  return train_loader








