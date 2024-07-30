#!/usr/bin/env python
# coding: utf-8

# In[96]:


#Name: 1) BHAWANI MAHTO(22030460016) 2) AYUSH RANJAN(22030460014) 3)ASHISH MAHATO(22030460013) 4)ABHAY KUMAR(22030460002)
#___________________________________________________________________________________________________________________________
#topic: single-variable linear regression of semester gpa vs weekly hours of study
#___________________________________________________________________________________________________________________________

#importing neccesary libraries
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[120]:


#reading the data-set cdv file stored 
df=pd.read_csv("gpa_study_hours.csv")
df


# In[98]:


df=df[['study_hours','gpa']]
df


# In[99]:


x=df['study_hours']
y=df['gpa']


# In[100]:


import matplotlib.pyplot as plt
plt.scatter(x,y,color='red',marker='+')
plt.xlabel("study_hours(per week)")
plt.ylabel("gpa")


# In[101]:


#splitting the imported data-set into 60% training data and 40% testing data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=23)


# In[102]:


x_train


# In[103]:


y_train


# In[104]:


import numpy as np
x_train=np.array(x_train).reshape(-1,1)
x_train


# In[105]:


from sklearn.linear_model import LinearRegression


# In[106]:


lr=LinearRegression()


# In[107]:


lr.fit(x_train,y_train)


# In[108]:


c=lr.intercept_


# In[109]:


c


# In[110]:


m=lr.coef_
m


# In[111]:


y_pred_train=m*x_train+c
y_pred_train


# In[112]:


y_pred_train1=lr.predict(x_train)


# In[113]:


y_pred_train1


# In[114]:


# trained model on train data(x_train,y_train)
import matplotlib.pyplot as plt
plt.scatter(x_train,y_train,color='green',marker='.')
plt.plot(x_train,y_pred_train1,color='red')
plt.xlabel("study_hours(per week)")
plt.ylabel("gpa")


# In[115]:


import numpy as np
x_test=np.array(x_test).reshape(-1,1)


# In[116]:


y_pred_test1=lr.predict(x_test)
y_pred_test1


# In[117]:


#testing the trained model on test data(x_test,y_test)
import matplotlib.pyplot as plt
plt.scatter(x_test,y_test,color='green',marker='+')
plt.plot(x_test,y_pred_test1,color='red')
plt.xlabel("study_hours(per week)")
plt.ylabel("gpa")


# In[118]:


gpa=lr.predict([[0]])


# In[119]:


gpa


# In[ ]:


#__________________________thank-you____________________________________________


# In[ ]:




