#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report

# In[17]:


X, y = make_classification(n_samples=1000, n_features=5, n_classes=2, random_state=42)

# In[18]:


X

# # train and testing

# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Initialize and train the decision tree classifier

# In[20]:


clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# In[21]:


y_pred = clf.predict(X_test)
y_pred

# # Calculate the confusion matrix

# In[22]:


cm = confusion_matrix(y_test, y_pred)
cm

# In[23]:


plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Predicted Negative', 'Predicted Positive'], yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Heatmap')
plt.show()

# # Split the confusion matrix into TP, FP, FN, TN

# The ravel() method is used to flatten the confusion matrix into a 1D array.

# In[24]:


tn, fp, fn, tp = cm.ravel()

# In[25]:


print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Negatives (TN): {tn}")

# # Calculate various performance metrics

# In[26]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# In[27]:


print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# In[28]:


print(classification_report(y_test, y_pred))

# # Display ROC-Curve

# In[29]:


from sklearn.metrics import RocCurveDisplay

# In[30]:


RocCurveDisplay.from_predictions(y_test, clf.predict(X_test))
plt.plot([0,1],[0,1])
plt.show()
