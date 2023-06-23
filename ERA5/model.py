import torch
import torch.nn as nn
import torch.nn.functional as F

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


