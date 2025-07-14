#!/usr/bin/env python
# coding: utf-8

# In[11]:


from sklearn.preprocessing import MinMaxScaler
import numpy as np

# In[12]:


data = np.array([[1.0, 2.0, 3.0],
                 [4.0, 5.0, 6.0],
                 [7.0, 8.0, 9.0]])

# In[13]:


scaler = MinMaxScaler()

# In[14]:


data_transformed = scaler.fit_transform(data)
data_transformed

# In[15]:


scaler.fit(data)
data_transformed_separate = scaler.transform(data)

# In[16]:


print("Original Data:")
print(data)

print("\nTransformed Data using fit_transform:")
print(data_transformed)

print("\nTransformed Data using fit and transform separately:")
print(data_transformed_separate)


# # Fit and Transform Seperately

# In[17]:


scaler.fit(data)
data_transformed_separate = scaler.transform(data)

# In[18]:


print("Original Data:")
print(data)

print("\nTransformed Data using fit and transform separately:")
print(data_transformed_separate)
