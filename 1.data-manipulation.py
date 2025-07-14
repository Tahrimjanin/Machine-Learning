#!/usr/bin/env python
# coding: utf-8

# In[454]:


#IMPORT LIBRARY
import pandas as pd

# In[455]:


#Empty df
df = pd.DataFrame()
df

# In[456]:


df.info()


# In[457]:


city = ['Berlin','Dhaka','Zurich']

# In[458]:


city

# In[459]:


df= pd.DataFrame(city)
df

# In[460]:


type(df)

# In[461]:


type(city)

# In[462]:


row,col=df.shape

# In[463]:


row

# In[464]:


col

# In[465]:


df.shape

# In[466]:


df

# In[467]:


df = pd.DataFrame(city,columns=['Big_City_In_World'])
df

# In[468]:


city2 = ['Dortmund','Paris','Potsdam']
df2 = pd.DataFrame(zip(city, city2), columns=['City1', 'City2'])


# In[469]:


df2

# In[470]:


df.shape

# In[471]:


df2

# In[472]:


df2.shape

# In[473]:


city3 = [['Bavaria', 'Erlangen'], ['NRW', 'Dortmund']]
city3

# In[474]:


df4 = pd.DataFrame(city3, columns=['State','City'])
df4

# In[475]:


dict1 = {
    'city1': city,
    'city2': city2
}

# In[476]:


dict1

# In[477]:


df5 = pd.DataFrame(dict1)

# In[478]:


df5

# In[479]:


import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[480]:


df = pd.read_csv("/kaggle/input/screen-time-dataset/user_behavior_dataset.csv")
df.head()


# In[481]:


# Print the shape of df
df.shape

# In[482]:


rows, cols = df.shape

# In[484]:


# Print the values of df
print(df.values)

# In[ ]:


print(df.columns)

# In[485]:


print(df.index)

# In[487]:


df.head(3)

# In[494]:


df[['Device Model']]

# In[495]:


df[['Device Model','Operating System']].head()

# In[499]:


df['Age'].sort_values

# In[500]:


df.describe()

# In[502]:


#correlation
df.corr(numeric_only=True)

# In[508]:


import seaborn as sns
sns.heatmap(numeric_df.corr(), annot=True)


# In[510]:


df.head()

# In[516]:


df['Age'] > 30

# In[518]:


df[df['Age'] > 30]

# In[519]:


#subseting rows
df[df['Operating System'] == 'Android']

# In[520]:


#subseting rows
df[(df['Operating System'] == 'Android') | (df['Operating System'] == 'iOS')]

# In[521]:


#Subsetting using isin()
df[df['Operating System'].isin(['Android', 'iOS'])]
