# Normalizations

Here CIFAR 10 dataset has been experimented using same model but with different normalization techniques.
Following is the summary of outputs of the 3 Normalizations:

---------------------------------------------------------------------------------------------------------------------------------------------------------------------
&#x1F537;**Batch Normalization**&#x1F537;

### Files:
S8_BN.ipynb\
model.py\
utils.py

### Target:
* To experiment on the CIFAR 10 dataset
* To use the model in the format of C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10 
* To keep the parameter count less than 50000
* To ensure Max Epochs is 20
* To achieve accuracy > 70%

### Results:
* Parameters: 46876
* Epochs: 15
* Best Training Accuracy: 75.04
* Best Test Accuracy: 77.41
* Graphs:
![image](https://github.com/nanekja/tsai/assets/12238843/e0a4534a-b33a-49ac-91a8-0f7b828767f5)
* MisClassified Images
![image](https://github.com/nanekja/tsai/assets/12238843/d87c4d64-a375-4fd5-823f-edcf821c5725)


### Analysis:


---------------------------------------------------------------------------------------------------------------------------------------------------------------------
&#x1F537;**Layer Normalization**&#x1F537;

### Files:
S8_LN.ipynb\
model.py\
utils.py

### Target:
* To experiment on the CIFAR 10 dataset
* To use the model in the format of C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10 
* To keep the parameter count less than 50000
* To ensure Max Epochs is 20
* To achieve accuracy > 70%

### Results:
* Parameters: 46876
* Epochs: 15
* Best Training Accuracy: 74.73
* Best Test Accuracy: 73.76
* Graphs:


### Analysis:


---------------------------------------------------------------------------------------------------------------------------------------------------------------------
&#x1F537;**Group Normalization**&#x1F537;

### Files:
S8_GN.ipynb\
model.py\
utils.py

### Target:
* To experiment on the CIFAR 10 dataset
* To use the model in the format of C1 C2 c3 P1 C3 C4 C5 c6 P2 C7 C8 C9 GAP C10 
* To keep the parameter count less than 50000
* To ensure Max Epochs is 20
* To achieve accuracy > 70%

### Results:
* Parameters: 46876
* Epochs: 15
* Best Training Accuracy: 71.6
* Best Test Accuracy: 71.9
* Graphs:


### Analysis:


---------------------------------------------------------------------------------------------------------------------------------------------------------------------
