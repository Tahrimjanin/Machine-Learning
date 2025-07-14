#!/usr/bin/env python
# coding: utf-8

# In[22]:


from sklearn.feature_extraction.text import CountVectorizer

# In[23]:


dataset = [
    'Sirajganj is an important jute collection',
    'processing, and trade centre that has road',
    'rail, and river connections to major cities',
     
]

# In[24]:


cv = CountVectorizer()

# In[25]:


X = cv.fit_transform(dataset)

# In[26]:


cv.get_feature_names_out()


# In[27]:


X.toarray()
