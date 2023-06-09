&#x1F537;**ERA Session 6**&#x1F537;

#### **Part A**

##### **Mathematics behind Back Propagation**

![image](https://github.com/nanekja/tsai/assets/12238843/2423f84c-1331-41dc-8d3a-83a0d4f704b9)
A sample fully connected neural network with 1 input layer, 1 hidden layer and 1 output layer

![image](https://github.com/nanekja/tsai/assets/12238843/8e196cc3-676b-4f5b-b0f8-2ce9828a3fb9)

* h1 and h2 are the weighted sum of the inputs i1 and i2	
* a_h1 and a_h2 are resultant of activation function (in this case sigmoid) applied to h1 and h2 respectively	
* o1 and o2 are the weighted sum of the inputs a_h1 and a_h2	
* a_o1 and a_o2 are resultant of activation function (in this case sigmoid) applied to o1 and o2 respectively	
* E1 and E2 are the variances of the outputs a_o1 and a_o2 with respect to targets t1 and t2	
* Combining E1 and E2 gives total Error	




---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#### **Part B**

The code is split into the following 3 files:

* S5.ipynb - A notebook file containing the model execution and accuracies
* model.py - A python file containing the actual model definition and configuration
* utils.py - A python file containing the training and testing transformations
