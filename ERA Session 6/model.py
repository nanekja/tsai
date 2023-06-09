import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchsummary import summary
from utils import *


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) #input -? OUtput? RF
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(256, 512, 3)
        self.conv6 = nn.Conv2d(512, 1024, 3)
        self.conv7 = nn.Conv2d(1024, 10, 3)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv2(F.relu(self.conv1(x)))))
        x = self.pool2(F.relu(self.conv4(F.relu(self.conv3(x)))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = F.relu(self.conv7(x))
        x = x.view(-1, 10)
        return F.log_softmax(x)


# Data to plot accuracy and loss graphs
train_losses = []
test_losses = []
train_acc = []
test_acc = []

test_incorrect_pred = {'images': [], 'ground_truths': [], 'predicted_vals': []}

def train(model, device, train_loader, optimizer):
  model.train()
  # here we are using the train component of the model
  pbar = tqdm(train_loader)

  train_loss = 0
  correct = 0
  processed = 0

  for batch_idx, (data, target) in enumerate(pbar):
    # here above we are looping through each batch represented by batch id (like 4th batch, 5th batch, etc) and data is images of batch and target is the label  
    data, target = data.to(device), target.to(device)
    # since model is being trained on GPU, we need to send data and targets also to GPU. They can't be on CPU
    optimizer.zero_grad()
    # When we do back propagation, the gradients will be stored at one place and we need to initiate gradients as zero to begin withwhich is done in this step
    # Predict
    pred = model(data)

    # Calculate loss
    loss = F.nll_loss(pred, target)
    # Computing loss using negative likelyhood loss function. Comparision is made between predicted output with targets
    train_loss+=loss.item()

    # Backpropagation
    loss.backward()
    # The loss is sent to backpropagation and here the gradients are computed based on the loss
    optimizer.step()
    # Applying the gradients to the parameters
    correct += GetCorrectPredCount(pred, target)
    processed += len(data)

    pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

  train_acc.append(100*correct/processed)
  train_losses.append(train_loss/len(train_loader))


def test(model, device, test_loader):
    model.eval()
    # here we are involking the evaluation method of the model object
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # Here we are looping through the images and labels in the test dataset
            data, target = data.to(device), target.to(device)
            # since model is being trained on GPU, we need to send data and targets also to GPU. They can't be on CPU
            output = model(data)
            # we are sending model output to output variable
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            # Computing loss using negative likelyhood loss function. Comparision is made between predicted output with targets
            # We are also summing up batch loss
            correct += GetCorrectPredCount(output, target)
            # Summing up the correct predictions

    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))



def draw_graphs():
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")