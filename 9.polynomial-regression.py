#!/usr/bin/env python
# coding: utf-8

# In[10]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# # Import Linear dataset

# In[11]:


df1 = pd.read_csv('/kaggle/input/linear-data/linear_data.csv')
df1.head()

# In[12]:


x_linear = df1[['x']]
y_linear = df1[['y']]

# # Import Non-linear dataset

# In[13]:


df2 = pd.read_csv('/kaggle/input/nonlinear-data/nonlinear_data.csv')
df2.head()

# In[14]:


x_nonlinear = df2[['x']]
y_nonlinear = df2[['y']]

# # Visual Representation

# In[15]:


#linear
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(x_linear, y_linear, color='blue', label='Data with Linear Relationship')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Relationship')
plt.legend()

#non-linear
plt.subplot(1, 2, 2)
plt.scatter(x_nonlinear, y_nonlinear, color='green', label='Data with Non-linear Relationship')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Non-linear Relationship')
plt.legend()

plt.tight_layout()
plt.show()

# In[16]:


df1.corr() #linear

# In[17]:


df2.corr() #non linear

# # Linear

# In[18]:


from sklearn.linear_model import LinearRegression

# In[19]:


reg1 = LinearRegression()

# In[20]:


reg1.fit(x_linear, y_linear) # x= x_linear.reshape(-1, 1)

# In[21]:


reg1.score(x_linear , y_linear)

# In[22]:


plt.scatter(x_linear, y_linear, color='blue', label='Data with Linear Relationship')
plt.plot(x_linear, reg1.predict(x_linear), color='red', label='Best fit line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Relationship')
plt.legend()

# # Non Linear

# In[23]:


reg2 = LinearRegression()

# In[24]:


reg2.fit(x_nonlinear , y_nonlinear)

# In[25]:


reg2.score(x_nonlinear, y_nonlinear)

# In[26]:


plt.scatter(x_nonlinear, y_nonlinear, color='green', label='Data with Non-linear Relationship')
plt.plot(x_nonlinear, reg2.predict(x_nonlinear), color='red', label='Best fit line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Non-linear Relationship')
plt.legend()

# # Polynomial Regression

# In[27]:


from sklearn.preprocessing import PolynomialFeatures

# In[39]:


poly = PolynomialFeatures(degree=6) # polynomial regression with deg 2 
X_poly = poly.fit_transform(x_nonlinear)

# In[29]:


X_poly.shape

# In[30]:


reg_poly = LinearRegression()

# In[31]:


reg_poly.fit(X_poly, y_nonlinear)

# In[32]:


reg_poly.score(X_poly, y_nonlinear)

# In[33]:


plt.scatter(x_nonlinear, y_nonlinear, color='green', label='Data with Non-linear Relationship')
plt.plot(x_nonlinear, 2 * np.sin(x_nonlinear), color='orange', label='True Non-linear Relationship')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Non-linear Relationship')
plt.legend()

# # Seperate Train Test

# In[34]:


from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(X_poly,y_nonlinear, test_size=.30, random_state=1 )

# In[35]:


reg = LinearRegression()

# In[36]:


reg.fit(xtrain, ytrain)

# # testing score

# In[37]:


reg.score(xtest, ytest)

# # training score

# In[38]:


reg.score(xtrain, ytrain) 
