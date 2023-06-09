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


def return_dataset_images(train_loader, total_images):
  batch_data, batch_label = next(iter(train_loader)) 

  fig = plt.figure()

  for i in range(total_images):
    plt.subplot(3,4,i+1)
    plt.tight_layout()
    plt.imshow(batch_data[i].squeeze(0), cmap='gray')
    plt.title(batch_label[i].item())
    plt.xticks([])
    plt.yticks([])

#this is from tqdm block
def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()





