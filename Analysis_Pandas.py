#!/usr/bin/env python
# coding: utf-8

# Import the libraries

# import the libraries
# numpy
# matplotlib
# pandas
# seaborn
# 
# load the dataset
# 
# outlook of dataset
# 
# data manipulation
# 
# data cleaning
# 
# data visualize wrt features
# 
# correlation betwween features

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Import the dataset

# In[2]:


pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# In[3]:


loan= pd.read_csv("loan_data.csv")
loan


# In[4]:


loan.shape


# In[5]:


loan.info()


# In[6]:


loan.describe()


# In[7]:


loan.columns


# In[8]:


loan['Property_Area'].describe()


# In[9]:


loan.head(9)


# In[10]:


loan.tail(6)


# In[11]:


loan.isnull().sum()


# In[12]:


loan.hist(figsize=(8,8))
plt.show()


# In[13]:


loan['ApplicantIncome'].hist()
plt.show()


# In[14]:


loan.boxplot(column='ApplicantIncome')
plt.show()


# In[15]:


loan.boxplot(column='ApplicantIncome',by='Education')
plt.tight_layout()
plt.show()


# In[16]:


loan.isnull().sum()


# In[17]:


loan["Gender"]  


# In[18]:


loan["Gender"].value_counts()    


# In[19]:


loan["Married"].value_counts()


# In[20]:



loan_Gender=loan["Gender"].value_counts(normalize=True)
loan_Gender


# In[21]:


loan_Gender.plot(kind = 'bar', title = "Gender",width=0.2)


# Categorical Values

# In[22]:


plt.subplot(221)
loan["Education"].value_counts(normalize=True).plot(kind="bar",title="Education",figsize=(10,10))
plt.tight_layout(pad=0.5)
plt.subplot(222)
loan["Married"].value_counts(normalize=True).plot(kind="bar",title="Married")
plt.tight_layout(pad=0.5)
plt.subplot(223)
loan["Self_Employed"].value_counts(normalize=True).plot(kind="bar",title="Self_Employed")
plt.subplot(224)
loan["Loan_Status"].value_counts(normalize=True).plot(kind="bar",title="Loan_Status")
plt.tight_layout(pad=0.5)


# In[23]:



loan["Gender"].fillna(value='Male',inplace=True)


# In[24]:


loan["Married"].fillna(loan["Married"].mode()[0],inplace=True)


# In[25]:


loan["Self_Employed"].fillna(loan["Self_Employed"].mode()[0],inplace=True)


# In[26]:


loan["Loan_Status"].fillna(loan["Loan_Status"].mode()[0],inplace=True)


# In[27]:


loan.isnull().sum()


# Fill Numerical values

# In[28]:


loan["LoanAmount"].fillna(loan["LoanAmount"].mean(),inplace=True)


# In[29]:


loan["Loan_Amount_Term"].fillna(loan["Loan_Amount_Term"].mean(),inplace=True)


# In[30]:


loan.isna().sum()


# Ordinal values

# In[31]:


plt.subplot(221)
loan["Dependents"].value_counts(normalize=True).plot(kind="bar",title="Dependents",figsize=(10,10))
plt.tight_layout(pad=0.5)

plt.subplot(222)
loan["Property_Area"].value_counts(normalize=True).plot(kind="bar",title="Property_Area")
plt.tight_layout(pad=0.5)


# In[32]:


loan["Dependents"].fillna(loan["Dependents"].mode(),inplace=True)


# Numerical value

# In[33]:


plt.subplot(131)
loan['ApplicantIncome'].plot.box(figsize=(8,5))
plt.tight_layout(pad=0.5)

plt.subplot(132)
loan['LoanAmount'].plot.box()
plt.tight_layout(pad=0.5)

plt.subplot(133)
loan['CoapplicantIncome'].plot.box()
plt.tight_layout(pad=0.5)


# In[34]:


plt.subplot(121)
loan.Credit_History.value_counts(normalize = 'True').plot(kind = 'bar', title='Credit_History')
plt.tight_layout(pad=0.5)


plt.subplot(122)
loan.Loan_Amount_Term.value_counts(normalize = 'True').plot(kind = 'bar', title='Loan_Amount_Term')
plt.tight_layout(pad=0.5)


# In[35]:


loan["Credit_History"].fillna(value = 1,inplace=True)
loan.isna().sum()


# In[36]:


loan['Total_income'] = loan["ApplicantIncome"]+loan["CoapplicantIncome"]

loan['Total_income_log'] = np.log(loan["Total_income"])

loan.columns


# In[46]:


loan.drop(['ApplicantIncome','CoapplicantIncome','Total_income'],axis=1,inplace=True)


# In[47]:


loan['Gender'].value_counts()


# In[39]:


loan['Gender'][loan['Loan_Status']=='Y'].value_counts()


# In[48]:


sns.set(rc={'figure.figsize':(10,8)})

plt.subplot(221)
sns.countplot(data=loan,x="Gender", hue='Loan_Status')

plt.subplot(222)
sns.countplot(data=loan,x="Married", hue='Loan_Status')


plt.subplot(223)
sns.countplot(x="Property_Area", hue='Loan_Status', data=loan)


# In[49]:


loan['Property_Area'][loan['Loan_Status']=='Y'].value_counts()


# In[50]:


loan['Property_Area'][loan['Loan_Status']=='N'].value_counts()


# In[51]:


loan.groupby('Gender')['Property_Area'].value_counts()


# In[52]:


loan.groupby('Gender')['Total_income_log'].mean()


# In[54]:


correlation_mat = loan.corr()
correlation_mat

