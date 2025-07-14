#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[1]:


import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import warnings as wr
wr.filterwarnings('ignore')

# # Load the dataset

# In[2]:


data = pd.read_csv('Churn_Modelling.csv')
data.head()

# In[3]:


data.shape

# In[4]:


data.isnull().sum()

# # Necessary preprocessing

# In[5]:


data = data.drop(columns=['RowNumber', 'CustomerId', 'Surname'], axis=1)
data.head()

# In[6]:


data.shape

# # Encoding categorical variables

# In[7]:


le_gender = LabelEncoder()
data['Gender'] = le_gender.fit_transform(data['Gender'])  # Gender: Female -> 0, Male -> 1
data.head()

# In[8]:


data.columns

# In[9]:


data['Geography'].value_counts()

# In[10]:


ohe_geography = OneHotEncoder(drop='first')
geo_encoded = ohe_geography.fit_transform(data[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=ohe_geography.get_feature_names_out(['Geography']))
geo_encoded_df

# In[11]:


data = pd.concat([data.drop(columns=['Geography']), geo_encoded_df], axis=1)
data.head()

# In[12]:


data.shape

# In[13]:


data.info()

# # Necessary EDA (basic descriptive statistics)

# In[14]:


exited_counts = data['Exited'].value_counts()

plt.figure(figsize=(6, 4))  

plt.pie(exited_counts, 
        labels=exited_counts.index, 
        autopct='%1.1f%%',  
        startangle=90,      
        colors=['lightblue', 'red'], 
        explode = (0, 0.1), 
        shadow=True)      

plt.title('Distribution of Exited Customers')
plt.axis('equal')
plt.show()

# In[15]:


data.corr()

# In[16]:


data.describe()

# # Splitting features and target

# In[17]:


X = data.drop(columns=['Exited'], axis=1)
y = data[['Exited']]

# In[18]:


X.head()

# In[19]:


y.head()

# # Split dataset into 70% train and 30% test

# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# In[21]:


X_train.shape

# In[22]:


X_test.shape

# # Decision Tree

# In[23]:


clf = DecisionTreeClassifier()

# In[24]:


clf.fit(X_train, y_train)

# In[25]:


y_pred = clf.predict(X_test)
y_pred

# In[26]:


accuracy_score(y_test, y_pred)

# In[27]:


print(classification_report(y_test, y_pred))

# In[28]:


cm = confusion_matrix(y_test, y_pred)
cm

# In[29]:


cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], 
                     columns=['Predicted Negative', 'Predicted Positive'])

fig = px.imshow(cm_df, 
                text_auto=True, 
                color_continuous_scale="Viridis", 
                title="Confusion Matrix")

fig.update_layout(
    title={
        'text': "Confusion Matrix",
        'x': 0.5, 
        'xanchor': 'center' 
    },
    
    xaxis_title="Predicted Label",
    yaxis_title="True Label",
    coloraxis_showscale=True
)
fig.show()

# [Watch: Confusion Matrix Tutorial](https://www.youtube.com/watch?v=gVByQqLso-I)

# # Random Forest

# [Random Forest Classifier](https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestClassifier.html) <br>
# [Random Forest Regressor](https://scikit-learn.org/1.5/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

# In[31]:


from sklearn.ensemble import RandomForestClassifier

# In[32]:


clf_ran = RandomForestClassifier(n_estimators=155)

# In[33]:


clf_ran

# In[34]:


clf_ran.fit(X_train, y_train)

# In[35]:


ran_pred = clf_ran.predict(X_test)
ran_pred

# In[36]:


accuracy_score(ran_pred, y_pred)

# # Confusion Matrix

# In[37]:


cm_ran = confusion_matrix(y_test, ran_pred)
cm_ran

# In[38]:


cm_df = pd.DataFrame(cm_ran, index=['Actual Negative', 'Actual Positive'], 
                     columns=['Predicted Negative', 'Predicted Positive'])

fig = px.imshow(cm_df, 
                text_auto=True, 
                color_continuous_scale="Viridis", 
                title="Confusion Matrix")

fig.update_layout(
    title={
        'text': "Confusion Matrix",
        'x': 0.5, 
        'xanchor': 'center' 
    },
    
    xaxis_title="Predicted Label",
    yaxis_title="True Label",
    coloraxis_showscale=True
)

fig.show()

# In[ ]:



