#!/usr/bin/env python
# coding: utf-8

# # Label Encoder
# 
# 

# In[63]:


classes = ['ClassA', 'ClassB', 'ClassC', 'ClassD']

instances = ['ClassA', 'ClassB', 'ClassC', 'ClassD', 'ClassA', 'ClassB', 'ClassC', 'ClassD', 'ClassA', 'ClassB']#real input that do encoded

# In[64]:


#Encoded

label_to_int = {label: index for index, label in enumerate(classes)} # num for each lebel like A=0,B=1,C=2 ,where has A there lebel 0
encoded_labels = [label_to_int[label] for label in instances]

print("Encoded labels:", encoded_labels)

# In[65]:


#Decoded

int_to_label = {index: label for label, index in label_to_int.items()}
decoded_labels = [int_to_label[index] for index in encoded_labels]

print("Encoded labels:", encoded_labels)
print("Decoded labels:", decoded_labels)

# # Sklearn(easy)
# 

# In[66]:


from sklearn.preprocessing import LabelEncoder

# In[67]:


label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(instances)

print("Encoded labels:", encoded_labels)

# In[68]:


original_labels = label_encoder.inverse_transform(encoded_labels)

print("Encoded labels:", encoded_labels)
print("Original labels:", original_labels)

# # One hot encoder

# In[69]:


import pandas as pd

# In[70]:


data = {'Category': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C']}

# In[71]:


df = pd.DataFrame(data)
df.head()

# In[72]:


one_hot_encoded_df = pd.get_dummies(df, columns=['Category'])
one_hot_encoded_df

# In[73]:


one_hot_encoded_df = pd.get_dummies(df, columns=['Category'], prefix='Dummy')
one_hot_encoded_df

# In[74]:


one_hot_encoded_df = pd.get_dummies(df, columns=['Category'], prefix='Dummy',drop_first=True )
one_hot_encoded_df

# In[75]:


df.head()

# # Binary Encoder

# In[76]:


import pandas as pd
import category_encoders as ce

# In[77]:


data = {'Category': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C']}
df = pd.DataFrame(data)

# In[78]:


df.head()

# In[79]:


df.shape

# In[80]:


encoder = ce.BinaryEncoder(cols=['Category'], return_df=True)

# In[81]:


df_binary_encoded = encoder.fit_transform(df)
df_binary_encoded

# # Ordinal Encoder

# In[82]:


import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# In[83]:


data = [
    ['good'], ['bad'], ['excellent'], ['average'], 
    ['good'], ['average'], ['excellent'], ['bad'], 
    ['average'], ['good']
]

# In[84]:


data = pd.DataFrame(data=data, columns=['reviews'])
data.head()

# In[85]:


data.shape

# In[86]:


categories = [['bad', 'average', 'good', 'excellent']]

# In[87]:


categories

# In[88]:


encoder = OrdinalEncoder(categories=categories)

# In[89]:


encoded_data = encoder.fit_transform(data)
encoded_data

# In[90]:


decoded_data = encoder.inverse_transform(encoded_data)
decoded_data
