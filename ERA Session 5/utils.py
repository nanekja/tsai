import sys
import os
import torch
import torch.optim as optim
from torchvision import datasets, transforms

def get_device():
  '''
  This method returns the device in use.
  If cuda(gpu) is available it would return that, otherwise it would return cpu.
  '''
  use_cuda = torch.cuda.is_available()
  return torch.device("cuda" if use_cuda else "cpu")

def train_transforms():
    # Train data transformations
    train_transforms = transforms.Compose([
    transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
    transforms.Resize((28, 28)),
    transforms.RandomRotation((-15., 15.), fill=0),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
    ])
    return train_transforms

def test_transforms():
    # Test data transformations
    test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1407,), (0.4081,))
    ])
    return test_transforms



