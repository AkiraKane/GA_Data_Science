# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 19:51:16 2013

@author: alexandersedgwick
"""

import pandas as pd
df=pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/data/nytimes.csv')

df
df.describe()

df[:10]

df.head()
df.tail()

dfg = df[['Age','Impressions','Clicks']].groupby(['Age']).agg([mean])

df['log_impressions'] = df['Impressions'].apply(log)



def map_age_category(x):
    if x < 18:
        return '1'
    elif x < 25:
        return '2'
    elif x < 32:
        return '3'
    elif x < 45:
        return '4'
    else:
        return '5'

df['age_categories'] = df['Age'].apply(map_age_category)

#click through rate age, gender signed in (clicks over impressions)
#Pandas to csv function




import pandas as pd
from urllib import urlopen

df = pd.DataFrame()

for x in range(1,3):
    page = urlopen("http://stat.columbia.edu/~rachel/datasets/nyt"+str(x)+".csv")
    data = pd.read_csv(page)
    df = df.append(data)    
groups = df.groupby(['Age','Gender','Signed_In'])
ans = groups.Clicks.sum()/groups.Impressions.sum()
ans.to_csv('/users/alexandersedgwick/dropbox/development/ga/data/test_output_class.csv', header=True)

df['CTR'] = df['Clicks']/df['Impressions']


df2 = df

from pandas import *
from sklearn import *
from numpy import *

df = df.dropna()

df.to_records()


regr = linear_model.LinearRegression()
regr.fit(test['CTR'],test['Age'])
regr.coef_



df[:, :1]



df(df['CTR'].notnull()


test = df[np.isinf(df['Age'])]
test = df[np.isneginf(df['Age'])]
test = df[np.isnan(df['CTR'])]




# Load the diabetes dataset
diabetes = datasets.load_diabetes()

# Use only one feature
diabetes_X = diabetes.data[:, np.newaxis]
diabetes_X_temp = diabetes_X[:, :, 2]

# Split the data into training/testing sets
diabetes_X_train = diabetes_X_temp[:-20]
diabetes_X_test = diabetes_X_temp[-20:]







