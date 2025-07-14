#!/usr/bin/env python
# coding: utf-8

# In[297]:


import pandas as pd 
from matplotlib import pyplot as plt 


# In[298]:



df = pd.read_csv("/kaggle/input/nasdaq100-csv/nasdaq100.csv", sep=';')

# In[299]:


df.head()

# In[300]:


df.isnull().sum()

# In[301]:


#df.drop('Date', axis=1)
df = df.drop(columns = ['Date'])

# In[302]:


df.head()

# In[303]:


plt.scatter(df['Starting (USD)'] , df['Ending (USD)'])
plt.xlabel('Starting (USD)')
plt.ylabel('Ending (USD)')
plt.title('NASDAQ100 Stock Prices')

# In[304]:


x = df.drop('Ending (USD)', axis=1)
x.head()

# In[305]:


y = df[['Ending (USD)']]
y.head()

# # Linear Regression

# In[306]:


from sklearn.linear_model import LinearRegression

# In[307]:


reg = LinearRegression()

# In[308]:


x.mean()

# In[309]:


y.mean()

# In[310]:


plt.scatter(x.mean() , y.mean(), color='red')
plt.scatter(df['Starting (USD)'] , df['Ending (USD)'])
plt.xlabel('Starting (USD)')
plt.ylabel('Ending (USD)')
plt.title('NASDAQ100 Stock Prices')

# In[311]:


reg.fit(x, y) 

# In[312]:


m = reg.coef_
m

# In[313]:


c = reg.intercept_
c

# In[314]:


# y = mx + c
m*16700 + c


# In[315]:



reg.predict([[16700]])

# In[316]:


df['Predicted_y'] = reg.predict(x)
df.head()

# In[317]:



#plt.plot(x, df['Predicted_y'])
plt.plot(x, reg.predict(x))
plt.scatter(x.mean() , y.mean(), color='red')
plt.scatter(df['Starting (USD)'] , df['Ending (USD)'])
plt.xlabel('Starting (USD)')
plt.ylabel('Ending (USD)')
plt.title('NASDAQ100 Stock Prices')

# In[318]:


reg.predict([[16600]])

# In[319]:


df.head()

# # Loss and Gross Profit

# In[320]:


df['loss'] = df['Ending (USD)'] - df['Predicted_y']

df.head()

# In[321]:


#MSE and mae
from sklearn.metrics import mean_squared_error, mean_absolute_error
mse = mean_squared_error(df['Ending (USD)'], df['Predicted_y'])
mse

# In[322]:


mae = mean_absolute_error(df['Ending (USD)'], df['Predicted_y'])
mae

# In[323]:


sum(abs(df['loss'])) / len(x)

# In[324]:


reg.score(x,y)

# In[325]:


#plt.plot(x, df['Predicted_y'])
plt.plot(x, reg.predict(x))
plt.scatter(x.mean() , y.mean(), color='red')
plt.scatter(df['Starting (USD)'] , df['Ending (USD)'])
plt.scatter(df['Starting (USD)'] , reg.predict(x))
plt.xlabel('Starting (USD)')
plt.ylabel('Ending (USD)')
plt.title('NASDAQ100 Stock Prices')

# # R Squared Value

# In[326]:


reg.score(x,y)

# In[ ]:


from sklearn.metrics import r2_score
r2_score(y, reg.predict(x))
