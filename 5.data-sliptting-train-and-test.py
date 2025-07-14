#!/usr/bin/env python
# coding: utf-8

# In[36]:


import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


# In[37]:


X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)


# In[38]:


X

# # Create a Synthetic Dataset for Demonstration

# 

# In[39]:


pd.DataFrame(X)

# In[40]:


X_train, X_test, y_train ,y_test = train_test_split(X,y, test_size=0.30, random_state=42)

# In[41]:


X_train.shape

# In[42]:


X_test.shape

# In[43]:


y_train.shape

# In[44]:


y_test.shape
