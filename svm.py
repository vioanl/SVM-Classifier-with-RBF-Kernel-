#!/usr/bin/env python
# coding: utf-8

# In[1]:


# imports
import numpy as mp
import pandas as pd

from sklearn import svm  
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
from sklearn.metrics import accuracy_score

minmax_scaler = MinMaxScaler(feature_range=(0,1))


# In[2]:


#Διαβάζουμε μέσω της βιβλιοθήκης pandas το .csv αρχείο
covtype= pd.read_table('covtype.data.csv',  sep=',' , header=None)



# In[3]:


#χωρίζουμε τα δεδομένα σε (x_train, y_train )και (x_test, y_test)
train=covtype.loc[0:15119, : ]
test=covtype.loc[15120: , : ]

x_train = train.loc[ : , 0:53]
y_train = train.loc[ : , 54]

x_test = test.loc[ : , 0:53]
y_test = test.loc[ : , 54]


# In[4]:


# κανονικοποιούμε στο διάστημα [0,1] 
x_train_minmax=minmax_scaler.fit_transform(x_train)
x_test_minmax=minmax_scaler.fit_transform(x_test)


# In[5]:


# παίρνουμε δείγμα  (2000 samples)
train_sample=resample(train, n_samples=2000, random_state=0)
x_train_sample=train_sample.loc[ : , 0:53]
y_train_sample=train_sample.loc[ : , 54]
x_train_sample_minmax=minmax_scaler.fit_transform(x_train_sample)


# In[6]:


# 1.1 SVM μοντέλα 
#model1 : C=1, gamma=0.1

model1=svm.SVC(C=1, kernel='rbf', gamma=0.1, decision_function_shape='ovo')
scores = cross_val_score(model1, x_train_sample_minmax, y_train_sample, cv=10, scoring='accuracy')
scores.mean()


# In[7]:


# model2 : C=8, gamma=0.1

model2=svm.SVC(C=8, kernel='rbf', gamma=0.1, decision_function_shape='ovo')
scores = cross_val_score(model2, x_train_sample_minmax, y_train_sample, cv=10, scoring='accuracy')
scores.mean()


# In[8]:


# model3 : C=8, gamma=1

model3=svm.SVC(C=8, kernel='rbf', gamma=1, decision_function_shape='ovo')
scores = cross_val_score(model3, x_train_sample_minmax, y_train_sample, cv=10, scoring='accuracy')
scores.mean()


# In[9]:


# model4 : C=16, gamma=1

model4=svm.SVC(C=16, kernel='rbf', gamma=1, decision_function_shape='ovo')
scores = cross_val_score(model4, x_train_sample_minmax, y_train_sample, cv=10, scoring='accuracy')
scores.mean()


# In[21]:


# model5 : C=16, gamma=2

model5=svm.SVC(C=16, kernel='rbf', gamma=2, decision_function_shape='ovo')
scores = cross_val_score(model5, x_train_sample_minmax, y_train_sample, cv=10, scoring='accuracy')
scores.mean()


# In[10]:


# model6 : C=32, gamma=4

model6=svm.SVC(C=32, kernel='rbf', gamma=4, decision_function_shape='ovo')
scores = cross_val_score(model6, x_train_sample_minmax, y_train_sample, cv=10, scoring='accuracy')
scores.mean()


# In[11]:


# model7 : C=128, gamma=1

model7=svm.SVC(C=128, kernel='rbf', gamma=1, decision_function_shape='ovo')
scores = cross_val_score(model7, x_train_sample_minmax, y_train_sample, cv=10, scoring='accuracy')
scores.mean()


# In[12]:


# model8 : C=128, gamma=2

model8=svm.SVC(C=128, kernel='rbf', gamma=2, decision_function_shape='ovo')
scores = cross_val_score(model8, x_train_sample_minmax, y_train_sample, cv=10, scoring='accuracy')
scores.mean()


# In[14]:


# model9 : C=128, gamma=4

model9=svm.SVC(C=128, kernel='rbf', gamma=4, decision_function_shape='ovo')
scores = cross_val_score(model9, x_train_sample_minmax, y_train_sample, cv=10, scoring='accuracy')
scores.mean()


# In[13]:


# model10 : C=4096,  gamma=0,5

model10=svm.SVC(C=4096, kernel='rbf', gamma=0.5, decision_function_shape='ovo')
scores = cross_val_score(model10, x_train_sample_minmax, y_train_sample, cv=10, scoring='accuracy')
scores.mean()



# In[16]:


#1.2 το καλύτερο μοντέλο είναι το model
model7.fit(x_train_minmax, y_train)

y_pred=model7.predict(x_test_minmax)
accuracy_score(y_test, y_pred)


# In[17]:


#1.3
print(model7.support_vectors_.shape)


# In[ ]:





# In[ ]:




