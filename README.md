# Brackprop-Hyperparameters
Understanding and tuning hyperparameters of neural networks. 

[Prerequisite: (1) Theoretical understanding of cost functions, activation functions and partial derivatives. (2) Soli knowledge of how NN works. (3)Python programming. (4) Deep Learning alogrithms and technqiues.]

<br>
<p align="center">
<img src = "dataset/DNN.png" "dataset/DNN.png" width = "500" height = "300">
 </p>
<br>

On this code, we are tuning several hyperparameters and modifying functions. Both functions receive ''a'' (activation) and ''y'' (target output), which are from one data instance and represented by column vectors.
Fn() returns a scalar, while derivative() returns a column vector containing the cost derivative for each node in the output layer; no multiplication by the derivative of the activation function.The hyperparameters are 

### Functions implied

- ### Cost functions
This parameter specifies the cost function. Each one must be implemented as a class. 
The class should have two static functions: fn() executes the definition of the function to compute the cost in during evaluation, and derivative() executes the function's derivative to compute the error during learning.  
  
Parameter options -- QuadraticCost, CrossEntropy, Loglikelihood (LL cost function should really be used when 'act_output' (the activation function of the output layer) = 'Softmax'. )
  
 - ### Activation functions (act_hidden)
This parameter specifies the activation function for nodes on all hidden layers, but EXCLUDING the output layer. Each one must be implemented as a class. The class should have two functions: a static method fn() executes the definition of the function to compute the node activation value, and a class method derivative() executes the function's derivative to compute the error during learning.
 
Parameter options  -- Sigmoid, Tanh, ReLU, Softmax

- ### Activation functions (act_output)
This parameter specifies the activation function for nodes on the output layer. Tanh is selected as the activation function for the output layer:

  - Because the output value will be between 1 and -1 [instead of 1 and 0], the only cost function that goes with Tanh is the quadratic cost. Therefore, if the cost function was set to anything besides 'QuadraticCost', change/overwrite the cost function to QuadraticCost and print a warning (to the user, e.g. "Tanh only accepts 'QuadraticCost' cost function.  Changing to QuadraticCost").  This is implemented, in set_parameters().
 
  - In the network startup code "NN_network2.py", some code is added (in SGD()) that changes the dataset when the output layer's activation is Tanh, in particular to make the target y values to be 1 or -1 (instead of 1 or 0). 

  
- ### Regularization
This parameter specifies the regularization method. The selected method is applied to all hidden layers and the output layer. The regularization is relevant at two places in the backprop algorithm: During training, when weights are adjusted at the end of a mini-bath -- the function update_mini_batch(). During evaluation, when the cost is computed -- the function total_cost().
Both of those functions have the parameter lmbda, passed in from the function SGD(). Utilize its value in implementing the regularizations.

Parameter options  -- L1, L2
  
- ### Dropout percentage
This parameter specifies the percentage of dropout. The value is between 0 and 1.  For example, 0.4 means 40% of the nodes on a layer are dropped or made inaccessible. Dropout consists in randomly setting a fraction rate of units in a layer to 0 at each update during training time, which helps prevent overfitting. Assume the same dropout percentage is applied to all hidden layers. Dropout should not be applied to input or output layer. 
  - Dropout is applied during the training phase only.  No dropout is applied during the testing/evaluation phrase.
  - Use the same dropout nodes during one mini-batch.  That means you have to store which nodes were used/dropped somewhere else. Think about it and implement in your way.
  - Scale the output values of the layer.  

Then during the backward propagation phase, you apply the mask to the delta of a hidden layer (dh1). This is necessary because, since a dropout mask is applied as an additional multiplier function after the activation function, it essentially became a constant coefficient of the activation function (i.e., c*a(z)), and shows up in the derivative of that function --  Let f(z) = c*a(z), then f'(z) = c*a'(z).
  
Modified functions

- set_parameters()
- feedforward()
- backprop()
- update_mini_batch()
- total_cost()

Using the application code, you should be able to generate the results for each combinations as in the results folder using  "iris.csv" and the "iris-423.dat". For each experiment, train the network using "iris-train-1.csv" for 15 epochs and test it using "iris-test-1.csv".


For dropoutpercentage, use "iris4-20-7-3.dat".The results may vary for these experiments because dropout nodes are randomly (but probabilistically) chosen.
