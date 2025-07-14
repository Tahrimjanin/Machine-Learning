#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd

# In[10]:


data = {
'X': [1, 2, 3, 4, 5, 7, 10],
'Y': [2, 3, 5, 4, 6, 6, 8]
}

# In[11]:


data

# In[12]:


df = pd.DataFrame(data)
df.head()

# # Calculate Pearson correlation

# In[13]:


pearson_corr = df['X'].corr(df['Y'])
print("Pearson correlation coefficient:", pearson_corr)

# In[14]:


df.corr()
