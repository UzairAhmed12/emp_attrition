#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[5]:


pip install -U imbalanced-learn --user


# In[6]:


import imblearn


# In[7]:


data= pd.read_csv(r"C:\Users\Uzair\Desktop\archive (18)\WA_Fn-UseC_-HR-Employee-Attrition.csv")


# In[8]:


data.head()


# In[9]:


pd.set_option('display.max_columns',None)


# In[10]:


data.head()


# In[11]:


data.info()


# In[12]:


data.describe()


# In[13]:


print(data.duplicated().value_counts())
data.drop_duplicates(inplace=True)
print(len(data))


# In[14]:


data.isnull().sum()


# # Target Variable

# In[15]:


plt.figure(figsize=(15,5))
plt.rc("font", size=14)
sns.countplot(y='Attrition',data=data, palette='Set1')
plt.show()


# # Exploratory Data Analysis

# In[16]:


plt.figure(figsize=(12,5))
plt.rc("font", size=14)
sns.countplot(x='Department',hue='Attrition',data=data, palette='Set1', color='Green')
plt.title("Attrition with respect to Department")
plt.show()


# In[17]:


plt.figure(figsize=(12,5))

sns.countplot(x='EducationField',hue='Attrition',data=data, palette='Set1', color='Green')
plt.title("Attrition with respect to EducationField")
plt.xticks(rotation=45)
plt.show()


# In[18]:


plt.figure(figsize=(12,5))

sns.countplot(x='JobRole',hue='Attrition',data=data, palette='Set1')
plt.title("JobRole with respect to Attrition")
plt.legend(loc='best')
plt.xticks(rotation=45)
plt.show()


# In[19]:


plt.figure(figsize=(12,5))
sns.countplot(x='Gender',hue='Attrition',data=data, palette='Set1')
plt.title("Gender with respect to Attrition")
plt.legend(loc='best')
plt.show()


# In[20]:


plt.figure(figsize=(12,5))
sns.distplot(data['Age'],hist=False)
plt.show()


# In[21]:


ordinal_features = ['Education','EnvironmentSatisfaction','JobInvolvement','JobSatisfaction',
                   'PerformanceRating','RelationshipSatisfaction','WorkLifeBalance']
data[ordinal_features].head()


# In[22]:


edu_map = {1 : 'Below College', 2: 'College', 3 : 'Bachelor', 4 :'Master', 5: 'Doctor'}
plt.figure(figsize=(12,5))
sns.countplot(x=data['Education'].map(edu_map), hue='Attrition', data=data, palette='Set1')
plt.title("Education with respect to Attrition")
plt.show()


# # Label Encoding

# In[23]:


data['Attrition'] = data['Attrition'].replace({'No':0,'Yes':1})


# In[24]:


data['OverTime'] = data['OverTime'].map({'No':0,'Yes':1})
data['Gender'] = data['Gender'].map({'Male':0,'Female':1})


# In[25]:


data["Gender"].value_counts()


# In[26]:


data['Over18'] = data['Over18'].map({'Y':1,'No':0})


# In[27]:


from sklearn.preprocessing import LabelEncoder
encoding_cols=['BusinessTravel','Department','EducationField','JobRole','MaritalStatus']
label_encoders = {}
for column in encoding_cols:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])
    


# In[28]:


data.head()


# In[29]:


data.info()


# In[33]:


X = data.drop(['Attrition'], axis=1)
y = data['Attrition'].values


# # Resampling

# In[31]:


from collections import Counter
from imblearn.over_sampling import RandomOverSampler
print(Counter(y))
rus = RandomOverSampler(random_state = 42)
X_over, y_over = rus.fit_resample(X,y)
print(Counter(y_over))


# In[34]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.2, random_state=42)


# In[36]:


print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# # Logestic Regression

# In[37]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, roc_auc_score


# In[39]:


logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# In[41]:


prediction = logreg.predict(X_test)
cnf_matrix = confusion_matrix(y_test,prediction)
print("Accuracy Score -", accuracy_score(y_test , prediction))


# In[42]:


fig = plt.figure(figsize = (15,6))
ax1 = fig.add_subplot(1,2,1)
ax1 = sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'Blues', fmt = 'd')
bottom, top = ax1.get_ylim()
ax1.set_ylim(bottom + 0.5, top - 0.5)
plt.xlabel('Predicted')
plt.ylabel('Expected')

ax2 = fig.add_subplot(1,2,2)
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = roc_curve(y_test, prediction)
auc = roc_auc_score(y_test, prediction)
ax2 = plt.plot(fpr,tpr,label = "data 1 auc="+str(auc))
plt.legend(loc=4)
plt.show()


# In[ ]:




