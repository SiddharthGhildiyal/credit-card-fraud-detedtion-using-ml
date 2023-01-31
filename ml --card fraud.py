#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
     
    


# In[ ]:


# loading the dataset to a Pandas DataFrame
credit_card_data = pd.read_csv('/content/credit_data.csv')


# In[ ]:


# first 5 rows of the dataset
credit_card_data.head()


# In[ ]:


credit_card_data.tail()


# In[ ]:


# dataset informations
credit_card_data.info()


# In[ ]:


# checking the number of missing values in each column
credit_card_data.isnull().sum()


# In[ ]:


# distribution of legit transactions & fraudulent transactions
credit_card_data['Class'].value_counts()


# In[ ]:


0    284315
1       492
Name: Class, dtype: int64


# In[ ]:


# separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[ ]:


print(legit.shape)
print(fraud.shape)


# In[ ]:


# statistical measures of the data
legit.Amount.describe()
     


# In[ ]:


fraud.Amount.describe()


# In[ ]:


# compare the values for both transactions
credit_card_data.groupby('Class').mean()


# In[ ]:


legit_sample = legit.sample(n=492)


# In[ ]:


new_dataset = pd.concat([legit_sample, fraud], axis=0)
     

new_dataset.head()
     


# In[ ]:


new_dataset.tail()


# In[ ]:


new_dataset['Class'].value_counts()
     
new_dataset.groupby('Class').mean()


# In[ ]:


X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']
     

print(X)


# In[ ]:


print(Y)


# In[ ]:


#now we will split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
     

print(X.shape, X_train.shape, X_test.shape)


# In[ ]:


model = LogisticRegression()
     

# training the Logistic Regression Model with Training Data
model.fit(X_train, Y_train)
     #we can alsouse lasso for this


# In[ ]:


# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
     

print('Accuracy on Training data : ', training_data_accuracy)


# In[ ]:


# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
     

print('Accuracy score on Test Data : ', test_data_accuracy)
     

