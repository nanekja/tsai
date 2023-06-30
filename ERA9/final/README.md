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
* Parameters: 184096
* Epochs: 100
* Best Training Accuracy: 74.79
* Best Test Accuracy: 85.2
* Graphs:
  ![image](https://github.com/nanekja/tsai/assets/12238843/45eac019-ba24-4e34-ae64-2e79c9395a5a)


### Analysis:

All the mentioned criteria listed under targets above is met. Implemented Dilated Convolution and as well as the Depthwise Convolution. didn't use MAx Pooling at all and have maintained the architecture and executed to achieve > 85% validation accuracy with < 200K Parameters.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------
