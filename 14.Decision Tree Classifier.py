#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# In[2]:


df = pd.read_csv('shop data.csv')

# In[3]:


df

# In[4]:


x = df.iloc[:,:-1]

# In[5]:


x

# In[6]:


y = df.iloc[:,4]

# In[7]:


y

# In[8]:


from sklearn.preprocessing import LabelEncoder

# In[9]:


le_x=LabelEncoder()
x = x.apply(LabelEncoder().fit_transform)

# In[10]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=.25,random_state=1)

# In[11]:


xtrain

# In[12]:


xtest

# In[13]:


from sklearn.tree import DecisionTreeClassifier

# In[14]:


dect = DecisionTreeClassifier()

# In[15]:


dect.fit(xtrain,ytrain)

# In[16]:


y_predict = dect.predict(xtest)

# In[17]:


from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(ytest,y_predict)

# In[18]:


cm

# In[19]:


xinput = np.array([1,0,0,1])

# In[ ]:


y_predict = dect.predict([xinput])

# In[ ]:


y_predict

# In[ ]:


import seaborn as sn
plt.figure(figsize = (10,7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')

# In[ ]:


dect.score(xtest,ytest)

# In[ ]:



