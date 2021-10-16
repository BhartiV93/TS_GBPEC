#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import the numpy and pandas package
import numpy as np
import pandas as pd
# Read the given CSV file, and view some sample records
advertising = pd.read_csv("Company_data.csv")
advertising


# In[2]:


# Info our dataset
advertising.info()

# Describe our dataset
advertising.describe()


# In[3]:


import matplotlib.pyplot as plt 
import seaborn as sns

# Using pairplot we'll visualize the data for correlation
sns.pairplot(advertising, x_vars=[ 'Radio','Newspaper'], 
             y_vars='Sales',height =5, kind='scatter')
plt.show()


# In[4]:


advertising.plot(kind="scatter",x='TV',y='Sales')
plt.show()


# In[5]:


advertising.corr()


# sales = c + m *TV

# Performing Simple Linear Regression
# 
# Equation of simple linear regression
# y = c + mX
# 
# For this dataset:
# 
# y = c + m * TV              m=coefficeient ,c = intercept

# 1.) Create Features(X) and Output(y)
# 
# 2.) Create Train and Test set
# 
# 3.)Train your model
# 
# 4.) Evaluate the model

# In[6]:


X = advertising['TV']      #feature
y = advertising['Sales']   #output


# In[7]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, 
                                                    test_size = 0.3, random_state = 100)


# Training and Testing datset

# In[8]:


X_train
y_train


# In[9]:


# Shape of the train set without adding column
X_train.shape

# Adding additional column to the train and test data
X_train = X_train.values.reshape(-1,1)
X_test = X_test.values.reshape(-1,1)

print(X_train.shape)
print(X_test.shape)


# In[16]:


from sklearn.linear_model import LinearRegression

# Creating an object of Linear Regression
lr = LinearRegression()

# Fit the model using .fit() method
lr.fit(X_train, y_train)


# In[17]:


# Intercept value
print("Intercept :",lr.intercept_)

# Slope value
print('Slope :',lr.coef_)


# In[12]:


from sklearn.metrics import r2_score


# In[18]:


# Making Predictions of y_value
y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)
# Comparing the r2 value of both train and test data
print(r2_score(y_train,y_train_pred))
print(r2_score(y_test,y_test_pred))


# In[14]:


from sklearn.metrics import mean_absolute_error
# The mean absolute error
print('Mean Absolute Error: %2f'
      % mean_absolute_error(y_test, y_test_pred))


# In[20]:


#Visualize the line on the test set
plt.scatter(X_test, y_test)
plt.plot(X_test, y_test_pred, 'r')
plt.show()

