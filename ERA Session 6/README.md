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

* To calculate the gradient of the error with respect to w5   :						`#ffffff` ∂E_total/∂w5 = ∂(E1 + E2)/∂w5 
* The partial derivative of E2 with respect to w5 is 0, hence :						∂E_total/∂w5 = ∂E1/∂w5	
* The partial derivative of E1 can be expanded as             :						∂E_total/∂w5 = ∂E1/∂a_o1*∂a_o1/∂o1*∂o1/∂w5	
* Calculating partial derivative of E1 with a_o1              :						∂E1/∂a_o1 =  ∂(½ * (t1 - a_o1)²)/∂a_o1 = (a_01 - t1)	
* Partial derivative of σ is σ(1-σ) and hence                 :						∂a_o1/∂o1 =  ∂(σ(o1))/∂o1 = a_o1 * (1 - a_o1)	
* Calculating partial derivative of o1 with w5                :						∂o1/∂w5 = a_h1	




---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#### **Part B**

The code is split into the following 3 files:

* S5.ipynb - A notebook file containing the model execution and accuracies
* model.py - A python file containing the actual model definition and configuration
* utils.py - A python file containing the training and testing transformations
