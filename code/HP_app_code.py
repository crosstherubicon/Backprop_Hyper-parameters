#!/usr/bin/env python
# coding: utf-8

# ## Hyperparameters application code

# In[1]:


import NN_network2 as network2
import numpy as np

# Test with one-data Iris data

inst1 = (np.array([5.7, 3, 4.2, 1.2]), np.array([0., 1., 0.]))
x1 = np.reshape(inst1[0], (4, 1))
y1 = np.reshape(inst1[1], (3, 1))
sample1 = [(x1, y1)]
inst2 = (np.array([4.8, 3.4, 1.6, 0.2]), np.array([1., 0., 0.]))
x2 = np.reshape(inst2[0], (4, 1))
y2 = np.reshape(inst2[1], (3, 1))
sample2 = [(x2, y2)]

net4 = network2.load_network("iris-423.dat")
net4.set_parameters(cost=network2.QuadraticCost)

net4.SGD(sample1, 2, 1, 1.0, evaluation_data=sample2, monitor_evaluation_cost=True, 
            monitor_evaluation_accuracy=True,
            monitor_training_cost=True,
            monitor_training_accuracy=True)


# ## Load the iris_train, iris_test datasets

# In[2]:


# Load the iris train-test (separate) data files
def my_load_csv(fname, no_trainfeatures, no_testfeatures):
    ret = np.genfromtxt(fname, delimiter=',')
    data = np.array([(entry[:no_trainfeatures],entry[no_trainfeatures:]) for entry in ret])
    temp_inputs = [np.reshape(x, (no_trainfeatures, 1)) for x in data[:,0]]
    temp_results = [np.reshape(y, (no_testfeatures, 1)) for y in data[:,1]]
    dataset = list(zip(temp_inputs, temp_results))
    return dataset

iris_train = my_load_csv('iris-train-1.csv', 4, 3)
iris_test = my_load_csv('iris-test-1.csv', 4, 3)


# ## (1) Sigmoid + Sigmoid + QuadraticCost 
# 

# In[5]:


net2 = network2.load_network("iris-423.dat")

# Set hyper-parameter values individually after the network
net2.set_parameters(cost=network2.QuadraticCost, act_hidden=network2.Sigmoid, act_output=network2.Sigmoid)

net2.SGD(iris_train, 15, 10, 1.0, evaluation_data=iris_test, monitor_evaluation_cost=True, 
            monitor_evaluation_accuracy=True,
            monitor_training_cost=True,
            monitor_training_accuracy=True)


# In[ ]:




