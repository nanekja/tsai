&#x1F537;**Resnet**&#x1F537;

### Target:
Reproducing the results and model of current SOTA for training time - CIFAR10 dataset according to DawnBench


Write a customLinks to an external site. ResNet architecture for CIFAR10 that has the following architecture:\
PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]\
Layer1 -\
X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]\
R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] \
Add(X, R1)\
Layer 2 -\
Conv 3x3 [256k]\
MaxPooling2D\
BN\
ReLU\
Layer 3 -\
X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]\
R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]\
Add(X, R2)\
MaxPooling with Kernel Size 4\
FC Layer \
SoftMax\
Uses One Cycle Policy such that:\
Total Epochs = 24\
Max at Epoch = 5\
LRMIN = FIND\
LRMAX = FIND\
NO Annihilation\
Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)\
Batch size = 512\
Use ADAM, and CrossEntropyLoss\
Target Accuracy: 90%\

### Files:
S10.ipynb\
Other files part of https://github.com/nanekja/pytorch_utils \

### Results:
* Parameters: 6573120
* Epochs: 24
* Best Training Accuracy: 93.96
* Best Test Accuracy: 90.98
* Graphs:
  


### Analysis:

This test is performed to find out the maximum learning rate which can be used to start the training with. In the case of the One Cycle Learning rate policy it is used to determine the peak learning rate as per the algorithm. \

Method: \

Start of with a learning rate close to zero and with each iteration(400 to 500 iterations) increase the Learning rate in an exponential manner
Plot the corresponding loss value and pick the maximum learning rate afteer which the learning rate \


As seen in the above image the second bottom appears at around 0.001 therefore the max learning rate is 0.001 of One cycle learning rate policy.

### Results of training
Max accuracy acheived = 90.9808% in 24 epochs

---------------------------------------------------------------------------------------------------------------------------------------------------------------------
