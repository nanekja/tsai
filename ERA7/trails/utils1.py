import sys
import os
import torch
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm


def train_transforms():
  # Train data transformations
  train_transforms = transforms.Compose([
    # This is the Transform (T) part of the ETL process where we are transforming the data into tensors
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

def return_image_stats(train):
  # We'd need to convert it into Numpy! Remember above we have converted it into tensors already
  train_data = train.train_data
  train_data = train.transform(train_data.numpy())

  print('[Train]')
  print(' - Numpy Shape:', train.train_data.cpu().numpy().shape)
  print(' - Tensor Shape:', train.train_data.size())
  print(' - min:', torch.min(train_data))
  print(' - max:', torch.max(train_data))
  print(' - mean:', torch.mean(train_data))
  print(' - std:', torch.std(train_data))
  print(' - var:', torch.var(train_data))

def return_single_image(train_loader):
  dataiter = iter(train_loader)
  images, labels = next(dataiter)

  print(images.shape)
  print(labels.shape)

  # Let's visualize some of the images
  plt.imshow(images[0].numpy().squeeze(), cmap='gray_r')


def return_dataset_images(train_loader, total_images):
  batch_data, batch_label = next(iter(train_loader)) 

  fig = plt.figure()

  for i in range(total_images):
    plt.subplot(6,10,i+1)
    plt.tight_layout()
    plt.imshow(batch_data[i].squeeze(0), cmap='gray')
    plt.title(batch_label[i].item())
    plt.xticks([])
    plt.yticks([])

#this is from tqdm block
def GetCorrectPredCount(pPrediction, pLabels):
  return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def draw_graphs(train_losses, train_acc, test_losses, test_acc):
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  axs[0, 0].plot(t)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(train_acc)
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test_losses)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(test_acc)
  axs[1, 1].set_title("Test Accuracy")


