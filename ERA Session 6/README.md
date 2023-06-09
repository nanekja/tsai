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

![image](https://github.com/nanekja/tsai/assets/12238843/a5904f0b-565d-4f46-abb7-0772132e9f80)


* To calculate the gradient of the error with respect to w5   :						`∂E_total/∂w5 = ∂(E1 + E2)/∂w5` 
* The partial derivative of E2 with respect to w5 is 0, hence :						`∂E_total/∂w5 = ∂E1/∂w5`	
* The partial derivative of E1 can be expanded as             :						`∂E_total/∂w5 = ∂E1/∂a_o1*∂a_o1/∂o1*∂o1/∂w5`	
* Calculating partial derivative of E1 with a_o1              :						`∂E1/∂a_o1 =  ∂(½ * (t1 - a_o1)²)/∂a_o1 = (a_01 - t1)`	
* Partial derivative of σ is σ(1-σ) and hence                 :						`∂a_o1/∂o1 =  ∂(σ(o1))/∂o1 = a_o1 * (1 - a_o1)`	
* Calculating partial derivative of o1 with w5                :						`∂o1/∂w5 = a_h1`	

* From above, the expanded partial derivative of error with w5:						`∂E_total/∂w5 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h1`
* Similarly, the expanded partial derivative of error with w6 :						`∂E_total/∂w6 = (a_01 - t1) * a_o1 * (1 - a_o1) *  a_h2`
* Similarly, the expanded partial derivative of error with w7 :						`∂E_total/∂w7 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h1`
* Similarly, the expanded partial derivative of error with w8 :						`∂E_total/∂w8 = (a_02 - t2) * a_o2 * (1 - a_o2) *  a_h2`

* Writing the full expansion of E1 with respect to a_h1       :						`∂E1/∂a_h1 = ∂E1/∂a_o1 * ∂a_o1/∂o1 * ∂o1/∂w5 * ∂w5/∂a_h1`	
* Substituting the equations from above                       :						`∂E1/∂a_h1 = (a_01 - t1) * a_o1 * (1 - a_o1) * w5`	
* Similarly E2 with respect to a_h1 is                        :						`∂E2/∂a_h1 = (a_02 - t2) * a_o2 * (1 - a_o2) * w7`	
* Total error with respect to a_h1 is sum of above 2 eq's     :						`∂E_total/∂a_h1 = (a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7`	
* Similarily the total error with respect to a_h2 is          :						`∂E_total/∂a_h2 = (a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8`	
							
* Expression for total error with w1 is                       :						`∂E_total/∂w1 = ∂E_total/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w1`	
* Similarily total error with w2                              :						`∂E_total/∂w2 = ∂E_total/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w2`	
* Similarily total error with w3                              :						`∂E_total/∂w3 = ∂E_total/∂a_h2 * ∂a_h2/∂h2 * ∂h2/∂w3`	
							
* Total error with w1 after substitution is                   :						`∂E_total/∂w1 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i1`	
* Total error with w2 after substitution is                   :						`∂E_total/∂w2 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w5 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w7) * a_h1 * (1 - a_h1) * i2`	
* Total error with w3 after substitution is                   :						`∂E_total/∂w3 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i1`	
* Total error with w4 after substitution is                   :						`∂E_total/∂w4 = ((a_01 - t1) * a_o1 * (1 - a_o1) * w6 +  (a_02 - t2) * a_o2 * (1 - a_o2) * w8) * a_h2 * (1 - a_h2) * i2`	

![image](https://github.com/nanekja/tsai/assets/12238843/baab00f0-d5ce-41ec-844a-a7e35472a88a)


Loss when Learning Rate: 0.1\
![image](https://github.com/nanekja/tsai/assets/12238843/bb04500c-bb42-425a-9f2e-bdd64e5a699e)

Loss when Learning Rate: 0.2\
![image](https://github.com/nanekja/tsai/assets/12238843/e91fb5d6-7b3c-4d85-a419-a91a5464b935)

Loss when Learning Rate: 0.5\
![image](https://github.com/nanekja/tsai/assets/12238843/9fa769cb-24e9-4597-a786-3a02a7e61b8a)

Loss when Learning Rate: 0.8\
![image](https://github.com/nanekja/tsai/assets/12238843/3d8c9b9a-ad28-4ed9-9da1-ca064ee3dde4)

Loss when Learning Rate: 1\
![image](https://github.com/nanekja/tsai/assets/12238843/ecf7d17e-d5a5-4647-838e-dba2faa52b30)

Loss when Learning Rate: 2\
![image](https://github.com/nanekja/tsai/assets/12238843/718411d0-8ff2-4999-9ad6-4693c5d4e3e3)



---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#### **Part B**

Model for MNIST Dataset with following constraints:
99.4% validation accuracy
Less than 20k Parameters
Less than 20 Epochs
Have used BN, Dropout,
(Optional): a Fully connected layer, have used GAP

* S6.ipynb - A notebook file containing the model execution and accuracies

Model Parameters: 16,954
Epochs: 20
Validation Accuracy: 99.47%
![image](https://github.com/nanekja/tsai/assets/12238843/818b9667-6f8b-424e-a706-8a185e41f312)


