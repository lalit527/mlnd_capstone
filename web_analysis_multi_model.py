#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[26]:


import warnings
import scipy
from datetime import timedelta
from pylab import rcParams
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.tsatools import lagmat
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.style.use('fivethirtyeight')
warnings.filterwarnings('ignore')


# ## Load Data

# In[3]:


path = Path('../input')


# In[4]:


train = pd.read_csv(path/'train_1.csv')
train.head()


# In[5]:


train.shape


# In[6]:


train_flattened = pd.melt(train[list(train.columns[-50:]) + ['Page']], id_vars='Page', var_name='date', value_name='Visits')
train_flattened['date'] = train_flattened['date'].astype('datetime64[ns]')
train_flattened['weekend'] = ((train_flattened.date.dt.dayofweek) // 5 == 1).astype(float)


# In[7]:


train_flattened.head()


# ### Median by Page

# In[8]:


df_median = pd.DataFrame(train_flattened.groupby(['Page'])['Visits'].median())
df_median.columns = ['median']
df_mean = pd.DataFrame(train_flattened.groupby(['Page'])['Visits'].mean())
df_mean.columns = ['mean']

train_flattened = train_flattened.set_index('Page').join(df_mean).join(df_median)


# In[9]:


train_flattened.head()


# In[10]:


train_flattened.reset_index(drop=False,inplace=True)


# In[11]:


train_flattened['weekday'] = train_flattened['date'].apply(lambda x: x.weekday())

train_flattened['year'] = train_flattened.date.dt.year
train_flattened['month'] = train_flattened.date.dt.month
train_flattened['day'] = train_flattened.date.dt.day


# In[12]:


train_flattened.head()


# ### Visualization

# In[13]:


plt.figure(figsize=(50, 8))
mean_group = train_flattened[['Page', 'date', 'Visits']].groupby(['date'])['Visits'].mean()
plt.plot(mean_group)
plt.title('Time Series - Average')
plt.show()


# In[14]:


plt.figure(figsize=(50, 8))
mean_group = train_flattened[['Page', 'date', 'Visits']].groupby(['date'])['Visits'].median()
plt.plot(mean_group, color='r')
plt.title('Time Series - median')
plt.show()


# In[15]:


plt.figure(figsize=(50, 8))
std_group = train_flattened[['Page','date','Visits']].groupby(['date'])['Visits'].std()
plt.plot(std_group, color = 'g')
plt.title('Time Series - std')
plt.show()


# In[16]:


train_flattened['month_num'] = train_flattened['month']
train_flattened['month'].replace('11', '11 - November', inplace=True)
train_flattened['month'].replace('12', '12 - December', inplace=True)

train_flattened['weekday_num'] = train_flattened['weekday']
train_flattened['weekday'].replace(0, '01 - Monday', inplace=True)
train_flattened['weekday'].replace(1, '02 - Tuesday', inplace=True)
train_flattened['weekday'].replace(2, '03 - Wednesday', inplace=True)
train_flattened['weekday'].replace(3, '04 - Thursday', inplace=True)
train_flattened['weekday'].replace(4, '05 - Friday', inplace=True)
train_flattened['weekday'].replace(5, '06 - Saturday', inplace=True)
train_flattened['weekday'].replace(6, '07 - Sunday', inplace=True)


# In[17]:


train_flattened.head()


# In[18]:


train_group = train_flattened.groupby(["month", "weekday"])["Visits"].mean().reset_index()
train_group = train_group.pivot('weekday', 'month', 'Visits')
train_group.sort_index(inplace=True)


# In[19]:


train_group.head()


# In[20]:


sns.set(font_scale=3.5)

f, ax = plt.subplots(figsize=(50, 30))

sns.heatmap(train_group, annot=False, ax=ax, fmt='d', linewidths=2)
plt.title('Web Traffic Months across Weekdays')
plt.show()


# In[21]:


train_day = train_flattened.groupby(['month', 'day'])['Visits'].mean().reset_index()
train_day = train_day.pivot('day', 'month', 'Visits')
train_day.sort_index(inplace=True)
train_day.dropna(inplace=True)


# In[22]:


f, ax = plt.subplots(figsize=(50, 30))

sns.heatmap(train_day, annot=False, ax=ax, fmt='d', linewidths=2)
plt.title('Web Traffic Months across days')
plt.show()


# ### ML Approach

# In[23]:


times_series_means = pd.DataFrame(mean_group).reset_index(drop=False)
times_series_means['weekday'] = times_series_means['date'].apply(lambda x: x.weekday())
times_series_means['Date_str'] = times_series_means['date'].apply(lambda x: str(x))
times_series_means[['year', 'month', 'day']] = pd.DataFrame(times_series_means['Date_str'].str.split('-', 2).tolist(), columns = ['year','month','day'])
date_staging = pd.DataFrame(times_series_means['day'].str.split(' ', 2).tolist(), columns=['day', 'other'])
times_series_means['day'] = date_staging['day'] * 1
times_series_means.drop('Date_str',axis = 1, inplace =True)
times_series_means.head()


# In[28]:


times_series_means.reset_index(drop=True,inplace=True)

def lag_func(data, lag):
    lag = lag
    X = lagmat(data['diff'], lag)
    lagged = data.copy()
    for c in range(1, lag+1):
        lagged["lag%d" % c] = X[:, c-1]
    return lagged

def diff_creation(data):
    data["diff"] = np.nan
    data.ix[1:, "diff"] = (data.iloc[1:, 1].as_matrix() - data.iloc[:len(data)-1, 1].as_matrix())
    return data

df_count = diff_creation(times_series_means)

lag = 7
lagged = lag_func(df_count,lag)
last_date = lagged['date'].max()


# In[29]:


lagged.head()


# In[30]:


def train_test(data_lag):
    xc = ["lag%d" % i for i in range(1, lag+1)] + ['weekday'] + ['day']
    split = 0.7
    xt = data_lag[(lag+1):][xc]
    yt = data_lag[(lag+1):]['diff']
    isplit = int(len(xt) * split)
    x_train, y_train, x_test, y_test = xt[:isplit], yt[:isplit], xt[isplit:], yt[isplit:]
    return x_train, y_train, x_test, y_test, xt, yt

x_train, y_train, x_test, y_test, xt, yt = train_test(lagged)


# In[31]:


from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# In[33]:


def modelisation(x_tr, y_tr, x_ts, y_ts, xt, yt, model0, model1):
    model0.fit(x_tr, y_tr)
    prediction = model0.predict(x_ts)
    r2 = r2_score(y_ts.as_matrix(), model0.predict(x_ts))
    mae = mean_absolute_error(y_ts.as_matrix(), model0.predict(x_ts))
    print ("-----------------------------------------------")
    print ("mae with 70% of the data to train:", mae)
    print ("-----------------------------------------------")

    model1.fit(xt, yt) 
    return model1, prediction, model0

model0 =  AdaBoostRegressor(n_estimators = 5000, random_state = 42, learning_rate=0.01)
model1 =  AdaBoostRegressor(n_estimators = 5000, random_state = 42, learning_rate=0.01)

clr, prediction, clr0 = modelisation(x_train, y_train, x_test, y_test, xt, yt, model0, model1)


# In[ ]:




