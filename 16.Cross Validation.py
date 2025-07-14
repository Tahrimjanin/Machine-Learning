#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


import numpy as np
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
import warnings
warnings.filterwarnings('ignore')

# # Generating synthetic data & splitting into train-test

# In[ ]:


X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# # XGBoost Classifier

# In[ ]:


clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
clf.fit(X_train, y_train)

# In[ ]:


clf.score(X_test,y_test)

# # Perform k-Fold Cross-Validation

# In[ ]:


kf = KFold(n_splits=5, random_state=42, shuffle=True)
kf_scores = cross_val_score(clf, X, y, cv=kf)

# In[ ]:


kf_scores

# # Perform Stratified k-Fold Cross-Validation

# In[ ]:


skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
skf_scores = cross_val_score(clf, X, y, cv=skf)

# In[ ]:


skf_scores

# In[ ]:



