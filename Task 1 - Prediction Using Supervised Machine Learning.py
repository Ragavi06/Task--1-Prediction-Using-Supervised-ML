#!/usr/bin/env python
# coding: utf-8

# # GRIP - THE SPARKS FOUNDATION
# 
# # Data Science & Business Analytics Internship
# 
# # Name : Ragavi Natarajan

# # TASK 1 - Prediction Using Supervised ML

# # Importing necessary libraries

# In[1]:


import pandas as pd # Data Manipulation
import numpy as np #  Data Manipulation
import matplotlib.pyplot as plt # Data Visualization
import seaborn as sns # Data Visualization


# # Reading the data

# In[37]:


data = pd.read_excel("study of hours.xlsx") #importing data from excel to python
data.head() # To read first 5 enties in the DF


# In[38]:


data.shape # size of Dataframe, it shows no. of rows and columns in a data set


# In[39]:


data.describe() # summary statistics


# In[40]:


data.info() # To find datatyes & missing values if any in the DF


# # Visualize the data set

# In[41]:


sns.scatterplot(x=data.Hours, y=data.Scores,color = 'r',marker ='+')
plt.title('Hours Vs Scores')
plt.xlabel('Hours')
plt.ylabel('Scores')


# The above graph showing the positive linear relation between the two variables Hours and Scores i.e., if hours increases scores also increases

# # Fitting the regression line

# In[42]:


sns.regplot(x=data.Hours, y=data.Scores,color = 'r',marker ='+')



# # Seperating the data set into independent(predictor) and dependent(target) variable

# In[43]:


X= data[['Hours']]
y=data['Scores']


# # Splitting the data into Train and Validation  

# In[44]:


from sklearn.model_selection import train_test_split
train_X, val_X, train_y,val_y = train_test_split(X, y, random_state = 0)


# # Building the ML Model

# In[45]:


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()


# In[46]:


#train the model
regressor.fit(train_X, train_y) 


# In[47]:


pred_y = regressor.predict(val_X) #predicting for test dataset


# In[48]:


pd.DataFrame({'Actual':val_y,'Predicted':pred_y}) #Comparing Actual and Predicted


# In[49]:


print('Train Acccuracy: ', regressor.score(train_X, train_y), '\nTest Accuracy :', regressor.score(val_X, val_y))


# In[50]:


#Testing with custom data of 9.25 rs per day
h = [[9.25]]
s = regressor.predict(h)
print('A student who studies ', h[0][0], ' hours is estimated to score ', s[0])


# # Evaluate the model

# In[51]:


#Mean absolute error
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error (val_y,pred_y))


# In[52]:


#max error
print('Max Error: ',metrics.max_error(val_y,pred_y))


# In[53]:


#Mean Squared Error
print('Mean Squared Error:', metrics.mean_squared_error(val_y,pred_y))


# # Thank You

# In[ ]:




