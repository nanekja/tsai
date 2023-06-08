import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3) 
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(256*4*4, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):                                 # Channel Size | RF | Jump Parameter
        x = F.relu(self.conv1(x))                         # 28 > 26 | 1>3 | 1>1
        x = F.max_pool2d(F.relu(self.conv2(x)),2)         # 26 > 24 > 12 | 3>5>6 | 1>1>2
        x = F.relu(self.conv3(x))                         # 12>10 | 6>10 | 2>2
        x =F.max_pool2d(F.relu(self.conv4(x)),2)          # 10>8>4 | 10>14>16 | 2>2>4
        x = x.view(-1, 4096)                              # 4*4*256 = 4096

        x = F.relu(self.fc1(x))                           # 4*4*256 input layers -> 50 hidden layers

        x = self.fc2(x)                                   # 50 hidden layers -> 10 layers  

        
        return F.log_softmax(x, dim=1)
        # Applies a softmax followed by a logarithm.
        # While mathematically equivalent to log(softmax(x)), doing these two operations separately is slower and numerically unstable. 
        # This function uses an alternative formulation to compute the output and gradient correctly.

def model_summary(model,input_size):
    summary(model, input_size)

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
