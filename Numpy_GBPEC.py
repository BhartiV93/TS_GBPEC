#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np


# In[12]:


#scalar 
arr_s = np.array([5])
arr_s


# In[14]:


#vector array
arr_v = np.array([1,2,3,4,5])
arr_v


# In[16]:


two_d = np.array([[1,2,3,4,5],[9,10,11,4,5]])
two_d
print(two_d.ndim)


# In[18]:


three_d = np.array([[[1,2,3,4,5],[9,10,11,4,5]],[[1,2,3,4,5],[9,10,11,4,5]]])
three_d


# In[21]:


arr_1 = np.arange(8).reshape(2,2,2)
arr_1


# In[31]:


np_ones = np.zeros((6,6),dtype="int16")
np_ones


# In[26]:


arr_1 = np.arange(10).reshape(2,5)
arr_1


# In[30]:


linspace = np.linspace(1,10,5,dtype="int16")
linspace


# In[35]:


arr_new = np.array([1,'2',2.0,3,4])
arr_new


# In[39]:


#random number generator
np_rand = np.random.randint(100)
np_rand


# In[42]:


np_rand = np.random.rand()
np_rand


# In[46]:


np_rand = np.random.randint(100,size=(2,3))
np_rand


# In[47]:


np_rand = np.random.rand(5,5)
np_rand


# In[10]:


A = np.array([[1,2,3],[3,4,5],[6,7,8]])
B = np.array([[7,8,9],[10,11,12],[2,3,4]])
c = np.dot(A,B)
c


# In[13]:


ht = [1.2,2.3,4.9,5.1,5.2,5.4,5.5,5.6,5.6,5.8,5.9,6.0,6.1,6.2,6.5,7.1,14.5,23.2,40.2]
arr = np.array(ht)
arr


# In[14]:


print(np.min(arr))


# In[15]:


print(np.max(arr))


# In[16]:


print(np.mean(arr))


# In[17]:


print(np.median(arr))


# In[18]:


print(np.std(arr))


# In[20]:


Q1 = np.percentile(arr,25)
Q1


# In[22]:


Q3 = np.percentile(arr,75)
Q3


# In[24]:


IQR = Q3-Q1
IQR


# In[27]:


lower_limit = Q1-1.5*IQR
upper_limit = Q3 + 1.5*IQR
lower_limit,upper_limit


# In[28]:


n1 = arr[arr<lower_limit]
n1


# In[29]:


n2 = arr[arr>upper_limit]
n2


# In[54]:


final_arr = arr[(arr>lower_limit)&(arr<upper_limit)]
final_arr


# In[33]:


#sorting of data
a = np.array([[12,15],[10,1]])
a


# In[35]:


arr1 = np.sort(a)
arr1


# In[37]:


arr2 = np.sort(a,axis=1)
arr2


# In[39]:


arr3 = np.sort(a,axis=0)
arr3


# In[43]:


d= np.sort(a,axis=None)
d


# In[51]:


#array attribute
b = np.array([[1,2,3,4],[5,6,7,8]])
print(b)
print(b.ndim)
print(b.shape)
print(b.size)
print(b.dtype)
print(b.itemsize)


# In[53]:


b[0,1]


# In[55]:


final_arr[0:10]


# In[56]:


b[0:2,1:3]

