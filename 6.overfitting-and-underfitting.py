#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from mlxtend.evaluate import bias_variance_decomp



# In[2]:


# Seed for reproducibility
np.random.seed(0)


# In[3]:



# Dummy data (X: features, y: target with noise)
X = np.random.rand(100, 1) * 10  # X between 0 and 10
y = 2 * X.squeeze() + np.random.randn(100)  # y = 2X + noise

# In[4]:


# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# In[5]:


# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# In[6]:


# Bias-Variance Decomposition
mse, bias, variance = bias_variance_decomp( model, X_train, y_train, X_test, y_test, loss='mse', num_rounds=200, random_seed=1)



# In[7]:


# Output results
print("MSE (Mean Squared Error):", mse)
print("Bias^2:", bias)
print("Variance:", variance)
