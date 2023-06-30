&#x1F537;**Using Advanced Convolutions**&#x1F537;

### Target:
Here CIFAR 10 dataset has been experimented using the architecture to C1-C2-C3-C4-Output.

Following are some of the constraints:
* No MaxPooling
* At least 3 convolutions, where the last one has a stride of 2 in each convolution block
* Total RF must be more than 44
* one of the layers must use Depthwise Separable Convolution
* one of the layers must use Dilated Convolution
* use GAP (compulsory)
* use albumentation library and apply:
- horizontal flip
- shiftScaleRotate
- coarseDropout (max_holes = 1, max_height=16px, max_width=16, min_holes = 1, min_height=16px, min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
* achieve 85% accuracy in as many epochs as we want
* Total Params to be less than 200k

### Files:
S9.ipynb\
model.py\
utils.py

### Results:
* Parameters: 46876
* Epochs: 15
* Best Training Accuracy: 75.04
* Best Test Accuracy: 77.41
* Graphs:
  ![image](https://github.com/nanekja/tsai/assets/12238843/e0a4534a-b33a-49ac-91a8-0f7b828767f5)

* MisClassified Images:\
  ![image](https://github.com/nanekja/tsai/assets/12238843/d87c4d64-a375-4fd5-823f-edcf821c5725)


### Analysis:

The results of Batch Normalization is superior over other normalization techniques as the achieved accuracy is higher and has relatively faster convergence.
The Layer Normalization comes next in the order of efficiency as its accuracy is better than the Group Normalization.


---------------------------------------------------------------------------------------------------------------------------------------------------------------------