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


def get_mis_classified_byloader(model, device, data_loader):
    model.eval()
    missed_images = []  # will contain list of batches, for a given batch will return list of indices not predicted correctly
    # empty list will indicate no mis predictions
    pred_labels = []  # contains list of predicted labels by each batch
    data_images = []  # contains list of images by each batch for plotting
    target_labels = []  # contains list of target labels by batch
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            # missed_images.append(torch.where(torch.not_equal(pred.squeeze(),target).cpu()))
            misses = torch.where(torch.not_equal(pred.squeeze(), target))
            data_images.append(data[misses].cpu())
            target_labels.append(target[misses].cpu())
            pred_labels.append(pred[misses].cpu())

    pred_labels = [x.item() for item in pred_labels for x in item]
    target_labels = [x.item() for item in target_labels for x in item]
    data_images = [x for item in data_images for x in item]

    return data_images, pred_labels, target_labels


def plot_misclassified(image_data, targeted_labels, predicted_labels, classes, no_images):
    no_images = min(no_images, len(predicted_labels))

    figure = plt.figure(figsize=(12, 5))

    for index in range(1, no_images + 1):
        image = denormalize(image_data[index - 1]).numpy().transpose(1, 2, 0)
        plt.subplot(2, 5, index)

        plt.imshow(image)
        plt.tick_params(left=False, right=False, labelleft=False,
                        labelbottom=False, bottom=False)
        title = "Target:" + str(classes[targeted_labels[index - 1]]) + "\nPredicted:" + str(
            classes[predicted_labels[index - 1]])
        plt.title(title)

