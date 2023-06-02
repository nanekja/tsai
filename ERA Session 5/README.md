&#x1F537;**Usage**&#x1F537;

The code is split into the following 3 files:

* S5.ipynb - A notebook file containing the model execution and accuracies
* model.py - A python file containing the actual model definition and configuration
* utils.py - A python file containing the training and testing transformations



&#x1F537;**Model File (model.py)**&#x1F537;


https://github.com/partha-goswami/EVA8/blob/main/Session6/models/model.py

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

&#x1F537;**Notebook File (S5.ipynb)**&#x1F537;


https://github.com/partha-goswami/EVA8/blob/main/Session6/Session6_Advanced_Topics_Assignment_Solution.ipynb

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

&#x1F537;**Utils File (utils.py)**&#x1F537;


https://github.com/partha-goswami/EVA8/blob/main/Session6/models/model.py

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

&#x1F537;**Model Parameters and Final RF**&#x1F537;

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             320
            Conv2d-2           [-1, 64, 24, 24]          18,496
            Conv2d-3          [-1, 128, 10, 10]          73,856
            Conv2d-4            [-1, 256, 8, 8]         295,168
            Linear-5                   [-1, 50]         204,850
            Linear-6                   [-1, 10]             510
================================================================
Total params: 593,200
Trainable params: 593,200
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.67
Params size (MB): 2.26
Estimated Total Size (MB): 2.94

**Model Parameters**: 593,200

**Best Validation Accuracy Obtained**: 99.50 (10 epochs)

---------------------------------------------------------------------------------------------------------------------------------------------------------------------

&#x1F537;**Training Log**&#x1F537;

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 1
Train: Loss=0.0045 Batch_id=1874 Accuracy=93.32: 100%|██████████| 1875/1875 [05:32<00:00,  5.64it/s]
Test set: Average loss: 0.0377, Accuracy: 9866/10000 (98.66%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 2
Train: Loss=0.0028 Batch_id=1874 Accuracy=98.18: 100%|██████████| 1875/1875 [05:24<00:00,  5.78it/s]
Test set: Average loss: 0.0270, Accuracy: 9905/10000 (99.05%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 3
Train: Loss=0.0045 Batch_id=1874 Accuracy=98.58: 100%|██████████| 1875/1875 [05:25<00:00,  5.77it/s]
Test set: Average loss: 0.0187, Accuracy: 9938/10000 (99.38%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 4
Train: Loss=0.0010 Batch_id=1874 Accuracy=98.79: 100%|██████████| 1875/1875 [05:25<00:00,  5.77it/s]
Test set: Average loss: 0.0262, Accuracy: 9919/10000 (99.19%)

Adjusting learning rate of group 0 to 1.0000e-02.
Epoch 5
Train: Loss=0.0167 Batch_id=1874 Accuracy=98.97: 100%|██████████| 1875/1875 [05:25<00:00,  5.76it/s]
Test set: Average loss: 0.0171, Accuracy: 9944/10000 (99.44%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 6
Train: Loss=0.0003 Batch_id=1874 Accuracy=99.43: 100%|██████████| 1875/1875 [05:30<00:00,  5.68it/s]
Test set: Average loss: 0.0141, Accuracy: 9958/10000 (99.58%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 7
Train: Loss=0.0148 Batch_id=1874 Accuracy=99.53: 100%|██████████| 1875/1875 [05:21<00:00,  5.82it/s]
Test set: Average loss: 0.0138, Accuracy: 9957/10000 (99.57%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 8
Train: Loss=0.0047 Batch_id=1874 Accuracy=99.55: 100%|██████████| 1875/1875 [05:19<00:00,  5.86it/s]
Test set: Average loss: 0.0143, Accuracy: 9952/10000 (99.52%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 9
Train: Loss=0.0018 Batch_id=1874 Accuracy=99.50: 100%|██████████| 1875/1875 [05:17<00:00,  5.91it/s]
Test set: Average loss: 0.0144, Accuracy: 9954/10000 (99.54%)

Adjusting learning rate of group 0 to 1.0000e-03.
Epoch 10
Train: Loss=0.0031 Batch_id=1874 Accuracy=99.56: 100%|██████████| 1875/1875 [05:16<00:00,  5.93it/s]
Test set: Average loss: 0.0151, Accuracy: 9950/10000 (99.50%)


---------------------------------------------------------------------------------------------------------------------------------------------------------------------

&#x1F537;**Validation Loss & Accuracy Graph**&#x1F537;




---------------------------------------------------------------------------------------------------------------------------------------------------------------------

