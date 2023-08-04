# Custom Resnet on CIFAR10 using pytorch Lightening and GradCAM
This repository contains an application for CIFAR-10 classification using PyTorch Lightning. Image Classification is implemented using custom Resnet. The Application includes functionalities for missclassification and GradCam

## Requirements
* Use the Custom ResNet architecture for CIFAR10
* Uses One Cycle Policy
* Uses this transform -RandomCrop 32, 32 (after padding of 4) >> FlipLR >> Followed by CutOut(8, 8)
* Batch size = 512 and Use ADAM, and CrossEntropyLoss
* Target Accuracy: 90%

### General Requirements
Spaces app needs to include these features:

ask the user whether he/she wants to see GradCAM images and how many, and from which layer, allow opacity change as well
ask whether he/she wants to view misclassified images, and how many
allow users to upload new images, as well as provide 10 example images
ask how many top classes are to be shown (make sure the user cannot enter more than 10)

