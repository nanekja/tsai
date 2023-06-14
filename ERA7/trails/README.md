---------------------------------------------------------------------------------------------------------------------------------------------------------------------
&#x1F537;**Model 1**&#x1F537;

### Files:
m1.ipynb\
model1.py\
utils1.py

### Target:
* Get the set-up right
* Set Transforms
* Set Data Loader
* Set Basic Working Code
* Set Basic Training  & Test Loop

### Results:
* Parameters: 6.3M
* Best Training Accuracy: 99.99
* Best Test Accuracy: 99.31

### Analysis:
* Extremely Heavy Model for such a problem
* Model is over-fitting, but we are changing our model in the next step

---------------------------------------------------------------------------------------------------------------------------------------------------------------------
&#x1F537;**Model 2**&#x1F537;

### Files:
m2.ipynb\
model2.py\
utils2.py

### Target:

* Get the basic skeleton right. We will try and avoid changing this skeleton as much as possible.
* No fancy stuff

### Results:
* Parameters: 194k
* Best Train Accuracy: 99.08
* Best Test Accuracy: 98.85

### Analysis:
* The model is still large, but working. 
* We see some over-fitting

---------------------------------------------------------------------------------------------------------------------------------------------------------------------
&#x1F537;**Model 3**&#x1F537;

### Files:
m3.ipynb\
model3.py\
utils3.py

### Target:

* Make the model lighter

### Results:
* Parameters: 10.7K
* Best Train Accuracy: 98.85
* Best Test Accuracy: 98.58

### Analysis:
* The model is still large, but working. 
* We see some over-fitting

---------------------------------------------------------------------------------------------------------------------------------------------------------------------
&#x1F537;**Model 4**&#x1F537;

### Files:
m4.ipynb\
model4.py\
utils4.py

### Target:

* Add Batch-norm to increase model efficiency

### Results:
* Parameters: 10.9K
* Best Train Accuracy: 99.76
* Best Test Accuracy: 99.28

### Analysis:
* We have started to see over-fitting now
* Even if the model is pushed further, it won't be able to get to 99.4

---------------------------------------------------------------------------------------------------------------------------------------------------------------------
&#x1F537;**Model 5**&#x1F537;

### Files:
m5.ipynb\
model5.py\
utils5.py

### Target:

* Add Regularization, Dropout

### Results:
* Parameters: 10.9K
* Best Train Accuracy: 99.3
* Best Test Accuracy: 99.17

### Analysis:
* Regularization working.
* But with the current capacity, not possible to push it further.
* We are also not using GAP, but depending on a BIG-sized kernel

---------------------------------------------------------------------------------------------------------------------------------------------------------------------
&#x1F537;**Model 6**&#x1F537;

### Files:
m6.ipynb\
model6.py\
utils6.py

### Target:

* Add GAP and remove the last BIG kernel

### Results:
* Parameters: 6K
* Best Train Accuracy: 98.53
* Best Test Accuracy: 98.3

### Analysis:
* Adding Global Average Pooling reduces accuracy - WRONG
* We are comparing a 10.9k model with 6k model. Since we have reduced model capacity, a reduction in performance is expected.

---------------------------------------------------------------------------------------------------------------------------------------------------------------------
&#x1F537;**Model 7**&#x1F537;

### Files:
m7.ipynb\
model7.py\
utils7.py

### Target:

* Increase model capacity. Add more layers at the end. 

### Results:
* Parameters: 11.9K
* Best Train Accuracy: 99.19
* Best Test Accuracy: 99.02

### Analysis:
* The model still showing over-fitting, possibly DropOut is not working as expected! Wait yes! We don't know which layer is causing over-fitting. Adding it to a specific layer wasn't a great idea. 
* Quite Possibly we need to add more capacity, especially at the end. 
* Closer analysis of MNIST can also reveal that just at RF of 5x5 we start to see patterns forming. 
* We can also increase the capacity of the model by adding a layer after GAP!

---------------------------------------------------------------------------------------------------------------------------------------------------------------------
&#x1F537;**Model 8**&#x1F537;

