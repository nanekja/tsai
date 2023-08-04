# Custom Resnet on CIFAR10 using pytorch Lightening and GradCAM
This repository contains an application for CIFAR-10 classification using PyTorch Lightning. Image Classification is implemented using custom Resnet. The Application includes functionalities for missclassification and GradCam

## Requirements
* Use the Custom ResNet architecture for CIFAR10
* Uses One Cycle Policy
* Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
* Batch size = 512 and Use ADAM, and CrossEntropyLoss
* Target Accuracy: 80%

### General Requirements
Spaces app needs to include these features:

ask the user whether he/she wants to see GradCAM images and how many, and from which layer, allow opacity change as well
ask whether he/she wants to view misclassified images, and how many
allow users to upload new images, as well as provide 10 example images
ask how many top classes are to be shown (make sure the user cannot enter more than 10)

### Files:
S12.ipynb\
https://github.com/nanekja/tsai/tree/master/ERA12/final
Other files part of https://github.com/nanekja/pytorch_utils 

#### Model Summary
![image](https://github.com/nanekja/tsai/assets/12238843/fda7e582-de11-4485-912c-8c6e3278ada0)

#### Accuracy
![image](https://github.com/nanekja/tsai/assets/12238843/f811d39a-60ec-4a68-9247-3caee105394c)

#### 10 Mis-Classified images:
![image](https://github.com/nanekja/tsai/assets/12238843/0017110c-2fa3-44be-9bd0-a1e32f138cba)

#### GradCam Images
![image](https://github.com/nanekja/tsai/assets/12238843/93f73803-e1f0-4526-8a2d-33374ec0c2a2)

