import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid
import torchvision
from torch.utils.data import Dataset


class Cifar10SearchDataset(torchvision.datasets.CIFAR10):
    def __init__(self, root="./data/cifar10", train=True, download=True, transform=None):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label

def get_mean_and_std(exp_data):
    '''Calculate the mean and std for normalization'''
    print(' - Dataset Numpy Shape:', exp_data.shape)
    print(' - Min:', np.min(exp_data, axis=(0,1,2)) / 255.)
    print(' - Max:', np.max(exp_data, axis=(0,1,2)) / 255.)
    print(' - Mean:', np.mean(exp_data, axis=(0,1,2)) / 255.)
    print(' - Std:', np.std(exp_data, axis=(0,1,2)) / 255.)
    print(' - Var:', np.var(exp_data, axis=(0,1,2)) / 255.)
    return np.mean(exp_data, axis=(0,1,2)) / 255., np.std(exp_data, axis=(0,1,2)) / 255.

def unnormalize(img):
  img = img.numpy().astype(dtype=np.float32)
  
  for i in range(img.shape[0]):
    img[i] = (img[i]*std[i])+mean[i]
  
  return np.transpose(img, (1,2,0))

def imgshow(images, labels):
    num_classes = 10
    # display 10 images from each category. 
    class_names = ['airplane','automobile','bird','cat','deer',
                'dog','frog','horse','ship','truck']
    r, c = 10, 11
    n = 5
    fig = plt.figure(figsize=(15,15))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)
    for i in range(num_classes):
        idx = np.random.choice(np.where(labels[:]==i)[0], n)
        ax = plt.subplot(r, c, i*c+1)
        ax.text(-1.5, 0.5, class_names[i], fontsize=14)
        plt.axis('off')
        for j in range(1, n+1):
            plt.subplot(r, c, i*c+j+1)
            plt.imshow(unnormalize(images[idx[j-1]]), interpolation='none')
            plt.axis('off')
    plt.show()

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

    
def draw_graphs(train_losses, train_acc, test_losses, test_acc):
    t = [t_items.item() for t_items in train_losses]
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(t)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")

def convert_image_np(inp, mean, std):
    """Convert normalized tensor to numpy image for display.

    Args:
        inp (tensor): Tensor image
        mean(np array): numpy array of mean of dataset
        std(np array): numpy array of standard deviation of dataset

    Returns:
        np array: a numpy image
    """

    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def plot_misclassified(model, test_loader, classes, device, dataset_mean, dataset_std, no_misclf=20, plot_size=(4,5), return_misclf=False):
    """Plot the images are wrongly clossified by model

    Args:
        model (instance): torch instance of defined model (pre trained)
        test_loader (instace): torch data loader of testing set
        classes (dict or list): classes in the dataset
                if dict:
                    key - class id
                    value - as class name
                elif list:
                    index of list correspond to class id and name
        device (str): 'cpu' or 'cuda' device to be used
        dataset_mean (tensor or np array): mean of dataset
        dataset_std (tensor or np array): std of dataset
        no_misclf (int, optional): number of misclassified images to plot. Defaults to 20.
        plot_size (tuple): tuple containing size of plot as rows, columns. Defaults to (4,5)
        return_misclf (bool, optional): True to return the misclassified images. Defaults to False.

    Returns:
        list: list containing misclassified images as np array if return_misclf True
    """
    count = 0
    k = 0
    misclf = list()
  
    while count<no_misclf:
        img_model, label = test_loader.dataset[k]
        pred = model(img_model.unsqueeze(0).to(device)) # Prediction
        # pred = model(img.unsqueeze(0).to(device)) # Prediction
        pred = pred.argmax().item()

        k += 1
        if pred!=label:
            img = convert_image_np(
                img_model, dataset_mean, dataset_std)
            misclf.append((img_model, img, label, pred))
            count += 1
    
    rows, cols = plot_size
    figure = plt.figure(figsize=(cols*3,rows*3))

    for i in range(1, cols * rows + 1):
        _, img, label, pred = misclf[i-1]

        figure.add_subplot(rows, cols, i) # adding sub plot
        plt.title(f"Pred label: {classes[pred]}\n True label: {classes[label]}") # title of plot
        plt.axis("off") # hiding the axis
        plt.imshow(img, cmap="gray") # showing the plot

    plt.tight_layout()
    plt.show()
    
    if return_misclf:
        return misclf
