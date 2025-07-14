#!/usr/bin/env python
# coding: utf-8

# # Car risk predict and mannaul value

# In[136]:


import pandas as pd
import numpy as np
from sklearn import linear_model

# In[137]:


df = pd.read_csv('/kaggle/input/car-data/car data.csv')


# In[138]:


df

# In[139]:


df.experience

# In[140]:


exm_fit = df.experience.mean()

# In[141]:


exm_fit

# In[142]:


#for fillup null dataset
df.experience = df.experience.fillna(exm_fit)

# In[143]:


df

# In[144]:


reg = linear_model.LinearRegression()

# In[145]:


reg.fit(df[['speed','car_age','experience']],df.risk)

# In[146]:


reg.predict([[137,3,3.334615]]) #for predict risk 0utput %

# In[147]:


reg.coef_

# In[148]:


reg.intercept_

# In[149]:


#mannually calculate the risk output
-0.0302617*137 + -0.14473869*3 + -0.27965041*3.334615 +62.68381438116153

