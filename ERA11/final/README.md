&#x1F537;**GradCam on Resnet-18**&#x1F537;

### Target:

* To train using Resnet 18 for upto 20 epochs on the CIFAR10 dataset
* To show loss curves for test and train datasets
* To show a gallery of 10 misclassified images
* To use GradCam and output on 10 misclassified images.

### Files:
S11.ipynb\
Other files part of https://github.com/nanekja/pytorch_utils 

### Results:
* Parameters: 11,173,962
* Epochs: 20
* Best Training Accuracy: 86.04%
* Best Test Accuracy: 89.42%
* Graphs:
 ![image](https://github.com/nanekja/tsai/assets/12238843/f3cda075-5253-4b5e-9e91-d15abbf653cb)


### Analysis:

This test is performed to leverage GradCam and also to find out the maximum learning rate which can be used to start the training with. In the case of the One Cycle Learning rate policy it is used to determine the peak learning rate as per the algorithm. \

Method: 

Start of with a learning rate close to zero and with each iteration(400 to 500 iterations) increase the Learning rate in an exponential manner
Plot the corresponding loss value and pick the maximum learning rate afteer which the learning rate \

GradCam Misclassified images:
![image](https://github.com/nanekja/tsai/assets/12238843/6b45f141-b88f-47bb-a803-67bc858c8aa9)


As seen in the above image the second bottom appears at around 0.001 therefore the max learning rate is 0.001 of One cycle learning rate policy.

### Results of training
Max accuracy acheived = 90.98% in 20 epochs and GradCam usage was shown

---------------------------------------------------------------------------------------------------------------------------------------------------------------------
