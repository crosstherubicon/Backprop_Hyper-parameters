# Brackprop-Hyperparameters
Understanding hyperparameters of neural networks. 

In this code, we are tuning several hyperparameters and modifying functions. Both functions receive ''a'' (activation) and ''y'' (target output), which are from one data instance and represented by column vectors.
Fn() returns a scalar, while derivative() returns a column vector containing the cost derivative for each node in the output layer; no multiplication by the derivative of the activation function.The hyperparameters are 

- ### Cost functions
This parameter specifies the cost function. Each one must be implemented as a class. 
The class should have two static functions: fn() executes the definition of the function to compute the cost in during evaluation, and derivative() executes the function's derivative to compute the error during learning.  
  
Parameter options -- QuadraticCost, CrossEntropy, Loglikelihood (LL cost function should really be used when 'act_output' (the activation function of the output layer) = 'Softmax'. )
  
 - ### Activation functions
This parameter specifies the activation function for nodes on all hidden layers, but EXCLUDING the output layer. Each one must be implemented as a class. The class should have two functions: a static method fn() executes the definition of the function to compute the node activation value, and a class method derivative() executes the function's derivative to compute the error during learning.
 
Parameter options  -- Sigmoid, Tanh, ReLU, Softmax
  
- ### Regularization
This parameter specifies the regularization method. The selected method is applied to all hidden layers and the output layer. The regularization is relevant at two places in the backprop algorithm: During training, when weights are adjusted at the end of a mini-bath -- the function update_mini_batch(). During evaluation, when the cost is computed -- the function total_cost().

Parameter options  -- L1, L2
  
- Dropout rate
  
Modified functions

- set_parameters()
- feedforward()
- backprop()
- update_mini_batch()
- total_cost()

