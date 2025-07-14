#!/usr/bin/env python
# coding: utf-8

# # Row code of Normalization

# In[58]:


def min_max_scaling(data):
  min_val= min(data)
  max_val = max(data)
  scaled_data = [(x - min_val) / (max_val- min_val) for x in data]
  return scaled_data



# In[59]:


data =[1, 20, 30, 4, 5]

# In[60]:


scaled_data = min_max_scaling(data)

# In[61]:


print('Original Data:', data)
print("Scaled Data (raw):", scaled_data)

# # using sklearn 

# In[62]:


from sklearn.preprocessing import MinMaxScaler

# In[63]:


data ={'Feature1' : [1,5,10,4,5],
       'Feature2' : [6,7,8,10,10] }  

# In[64]:


df = pd.DataFrame(data)
df.head()

# In[65]:


scaler = MinMaxScaler()
scaler_data = scaler.fit_transform(df)
scaler_df= pd.DataFrame(scaler_data, columns=df.columns)

# In[66]:


print("original Dataframe:")
print(df)
print("\nScalered Dataframe:")
print(scaler_df)

# # Row code of standardization

# In[67]:


data = {
    'Feature 1': [1, 20, 3, 40, 5],
    'Feature 2': [6, 7, 18, 19, 10]
}

df = pd.DataFrame(data)

# In[68]:


df.head()

# In[69]:


def standardize_data(column):
    mean = column.mean()
    std_dev = column.std()
    standardized_data_raw = (column - mean) / std_dev
    return standardized_data_raw

# In[70]:


standardized_df_raw = df.apply(standardize_data)

# In[71]:


print("Original DataFrame:")
print(df)

print("\nStandardized DataFrame:")
print(standardized_df_raw)

# # using sklearn

# In[72]:


from sklearn.preprocessing import StandardScaler

# In[73]:


scaler = StandardScaler()
standardized_data = scaler.fit_transform(df)
standardized_df = pd.DataFrame(standardized_data, columns=df.columns)

# In[74]:


print("Original DataFrame:")
print(df)
print("\nStandardized DataFrame:")
print(standardized_df_raw)
print("\nStandardized DataFrame: sklearn")
print(standardized_df)

# 
