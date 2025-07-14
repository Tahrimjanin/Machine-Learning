#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

# In[2]:


# Generating linear data
x_linear = np.linspace(0, 10, 50)
y_linear = 2 * x_linear + 5 + np.random.randn(50) * 2  # adding some noise

# In[3]:


# Generating non-linear data
x_nonlinear = np.linspace(0, 10, 50)
y_nonlinear = 2 * np.sin(x_nonlinear) + np.random.randn(50)  # adding some noise

# In[4]:


#linear
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(x_linear, y_linear, color='blue', label='Data with Linear Relationship')
plt.plot(x_linear, 2 * x_linear + 5, color='red', label='True Linear Relationship')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Relationship')
plt.legend()

#non-linear
plt.subplot(1, 2, 2)
plt.scatter(x_nonlinear, y_nonlinear, color='green', label='Data with Non-linear Relationship')
plt.plot(x_nonlinear, 2 * np.sin(x_nonlinear), color='orange', label='True Non-linear Relationship')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Non-linear Relationship')
plt.legend()

plt.tight_layout()
plt.show()

# In[5]:


#correlation coefficients
correlation_linear = np.corrcoef(x_linear, y_linear)[0, 1]
correlation_nonlinear = np.corrcoef(x_nonlinear, y_nonlinear)[0, 1]

# In[6]:


print("Correlation coefficient for linear relationship:", correlation_linear)
print("Correlation coefficient for non-linear relationship:", correlation_nonlinear)

# In[7]:


from sklearn.linear_model import LinearRegression

# # Linear

# In[8]:


reg1 = LinearRegression()

# In[9]:


reg1.fit(x_linear.reshape(-1, 1) , y_linear) # x= x_linear.reshape(-1, 1)

# In[10]:


reg1.score(x_linear.reshape(-1, 1) , y_linear)

# In[11]:


plt.plot(x_linear, reg1.predict(x_linear.reshape(-1, 1)))

# In[12]:


plt.scatter(x_linear, y_linear, color='blue', label='Data with Linear Relationship')
plt.plot(x_linear, reg1.predict(x_linear.reshape(-1, 1)), color='green', label='Best fit line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Relationship')
plt.legend()

# # Non Linear

# In[13]:


reg2 = LinearRegression()

# In[14]:


reg2.fit(x_nonlinear.reshape(-1, 1) , y_nonlinear)

# In[15]:


reg2.score(x_nonlinear.reshape(-1, 1) , y_nonlinear)

# In[16]:


plt.plot(x_nonlinear, reg2.predict(x_nonlinear.reshape(-1, 1)))

# In[17]:


plt.scatter(x_nonlinear, y_nonlinear, color='green', label='Data with Non-linear Relationship')
plt.plot(x_nonlinear, 2 * np.sin(x_nonlinear), color='orange', label='True Non-linear Relationship')
plt.plot(x_nonlinear, reg2.predict(x_nonlinear.reshape(-1, 1)), color='red', label='Best fit line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Non-linear Relationship')
plt.legend()

#  

# Learn data science smartly: https://aiquest.org/
