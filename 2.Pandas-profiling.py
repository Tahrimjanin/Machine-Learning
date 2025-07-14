#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
#from pandas_profiling import ProfileReport
from ydata_profiling import ProfileReport

# In[2]:


url = 'https://raw.githubusercontent.com/rashakil-ds/Public-Datasets/refs/heads/main/credit.csv'
df = pd.read_csv(url)
df.head()

# In[3]:


df.shape

# # Generate a profiling report

# In[ ]:


profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
profile

# In[ ]:



