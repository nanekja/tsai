import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

from torch.utils.data import Dataset


def get_mean_and_std(exp_data):
    '''Calculate the mean and std for normalization'''
    print(' - Dataset Numpy Shape:', exp_data.shape)
    print(' - Min:', np.min(exp_data, axis=(0,1,2)) / 255.)
    print(' - Max:', np.max(exp_data, axis=(0,1,2)) / 255.)
    print(' - Mean:', np.mean(exp_data, axis=(0,1,2)) / 255.)
    print(' - Std:', np.std(exp_data, axis=(0,1,2)) / 255.)
    print(' - Var:', np.var(exp_data, axis=(0,1,2)) / 255.)
    return np.mean(exp_data, axis=(0,1,2)) / 255., np.std(exp_data, axis=(0,1,2)) / 255.


def plot_data(data, rows, cols):
    """Randomly plot the images from the dataset for vizualization

    Args:
        data (instance): torch instance for data loader
        rows (int): number of rows in the plot
        cols (int): number of cols in the plot
    """
    figure = plt.figure(figsize=(cols*2,rows*3))
    for i in range(1, cols*rows + 1):
        k = np.random.randint(0,50000)
        figure.add_subplot(rows, cols, i) # adding sub plot

        img, label = data[k]
        
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Class: {label} '+data.classes[label])

    plt.tight_layout()
    plt.show()

    