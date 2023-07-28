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

This test is performed to leverage GradCam and also to find out the maximum learning rate which can be used to start the training with. In the case of the One Cycle Learning rate policy it is used to determine the peak learning rate as per the algorithm. 

Method: 

Start of with a learning rate close to zero and with each iteration(400 to 500 iterations) increase the Learning rate in an exponential manner
Plot the corresponding loss value and pick the maximum learning rate afteer which the learning rate 

### 10 Mis-Classified images:
![image](https://github.com/nanekja/tsai/assets/12238843/0017110c-2fa3-44be-9bd0-a1e32f138cba)


### GradCam 10 Misclassified images:
![image](https://github.com/nanekja/tsai/assets/12238843/6b45f141-b88f-47bb-a803-67bc858c8aa9)
![image](https://github.com/nanekja/tsai/assets/12238843/1008d8b5-4e6e-43ca-b212-3498d7c09056)
![image](https://github.com/nanekja/tsai/assets/12238843/00361c5f-c1fa-421b-9812-bece53a68b84)
![image](https://github.com/nanekja/tsai/assets/12238843/b543c14a-ae0d-4041-b27b-76ad42054daa)
![image](https://github.com/nanekja/tsai/assets/12238843/dfe38448-31a6-4361-aae0-bd8c74e7baa8)
![image](https://github.com/nanekja/tsai/assets/12238843/b8913079-7abe-4337-9135-0a08646f5ab2)
![image](https://github.com/nanekja/tsai/assets/12238843/49be0cb5-51fc-4058-9ea0-0b9a468be200)
![image](https://github.com/nanekja/tsai/assets/12238843/d3bcbead-be72-401b-8ea9-24885530ff6b)
![image](https://github.com/nanekja/tsai/assets/12238843/2479c586-199f-4c0b-b754-5e46e4b15da2)
![image](https://github.com/nanekja/tsai/assets/12238843/502094dc-40ce-4851-a3af-22ac0486aaa8)


### Results of training
Max accuracy acheived = 89.42% in 20 epochs and GradCam usage was shown

---------------------------------------------------------------------------------------------------------------------------------------------------------------------
