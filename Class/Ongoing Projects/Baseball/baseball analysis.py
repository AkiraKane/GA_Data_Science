# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 08:52:06 2013
@author: asedgwick
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import *
from sklearn.feature_selection import SelectPercentile, f_classif

#Load the data
#baseball_train = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/ongoing projects/baseball/baseball_training_2011.csv')
#baseball_test = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/ongoing projects/baseball/baseball_test_2012.csv')

#Identify the different data types
# baseball_train.dtypes

#isolate only the most relevent inputs
#baseball_train = baseball_train[['playerID','lahmanID','birthDate','weight','height','bats','throws','college','yearID','teamID','lgID','stint','G','G_batting','AB','R','H','X2B','X3B','HR','RBI','SB','CS','BB','SO','IBB','HBP','SH','SF','GIDP','salary']]


#drop any records with NAs in the data I've identified
#baseball_train = baseball_train.dropna
#Run an inital regression model
#input = baseball_train[['HR','RBI','R','G','SB','height','weight']].values
#output = baseball_train['salary'].values
#lm1 = linear_model.LinearRegression()
#lm1.fit(input,output)
#lm1.coef_
#lm1.intercept_
#lm1.score(input,output)

#Explore features using histograms
#hist(baseball_train.salary)
#pd.scatter_matrix(pd.DataFrame(baseball_train[['HR','RBI','R','G','SB','height','weight']]), diagonal='kde', color='k', alpha=0.3)


#The following need to be converted because they are heavily skewed
#salary
#HR
#RBI
#R
#SB

baseball_train['log_salary'] = log(baseball_train.salary)

#baseball_train['log_HR'] = log(baseball_train.HR)
#baseball_train['log_RBI'] = log(baseball_train.RBI)
#baseball_train['log_R'] = log(baseball_train.R)
#baseball_train['log_SB'] = log(baseball_train.SB)
#input = baseball_train[['HR','RBI','R','G','SB','height','weight']].values
#output = baseball_train['log_salary'].values
#lm2 = linear_model.LinearRegression()
#lm2.fit(input,output)
#lm2.coef_
#lm2.intercept_
#lm2.score(input,output)
#lm2.fit
#
#
##This linear regression
#input = baseball_train[['HR','RBI','R','G','SB','height','weight']].values
#output = baseball_train['salary'].values
#import statsmodels.api as sm
#input = sm.add_constant(input)
#res = sm.OLS(output, input).fit()
##print res.params
##print res.bse
#print res.summary()
#print res.mse_model

#Benchmark?
import locale
locale.setlocale(locale.LC_ALL, 'en_US')
locale.format("%d",res.mse_total,grouping=True)


###Compare output to benchmark code
import statsmodels.api as sm
import locale
locale.setlocale(locale.LC_ALL, 'en_US')

b2011 = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/baseball/baseball_training_2011.csv')
b2012 = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/baseball/baseball_test_2012.csv')


#
#train_X = b2011[['G', 'AB', 'R', 'H', 'X2B', 'X3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF']].values
#train_y = b2011['salary'].values
#features = ['G', 'AB', 'R', 'H', 'X2B', 'X3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF']

train_X = b2011[['G', 'AB', 'R', 'H', 'X2B', 'X3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF']]
train_y = b2011['salary']



#
test_X = b2012[['G', 'AB', 'R', 'H', 'X2B', 'X3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF']].values
b2012_csv = b2012[['playerID','yearID', 'salary']]

#
input = train_X
output = train_y
input = sm.add_constant(input)

res = sm.OLS(output, input).fit() 
res.summary() 
print 'MSE:',locale.format("%d",metrics.mean_squared_error(res.predict(input), output),grouping=True)

#Test for multicollinearity
# Feature Selection!
features_select = feature_selection.f_regression(input,output,center=True)

for i in range(len(features)):
    print "p-value for ", features[i], ":", feature_selection.f_regression(input,output,center=True)[1][i]

#Correlation matrix

np.corrcoef(input,input)
#Explore features using histograms
hist(output)
#pd.scatter_matrix(pd.DataFrame(b2011[['G', 'AB', 'R', 'H', 'X2B', 'X3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF']]), diagonal='kde', color='k', alpha=0.3)
pd.scatter_matrix(input,diagonal='kde', color='k', alpha=0.3)





#Alex's version

import statsmodels.api as sm
import locale
locale.setlocale(locale.LC_ALL, 'en_US')

b2011 = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/baseball/baseball_training_2011.csv')
b2012 = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/baseball/baseball_test_2012.csv')


train_X = b2011[['teamID','G', 'AB', 'R', 'H', 'X2B', 'X3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF']]
train_y = b2011['salary']



#
test_X = b2012[['G', 'AB', 'R', 'H', 'X2B', 'X3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF']].values
b2012_csv = b2012[['playerID','yearID', 'salary']]

#
input = train_X
output = train_y
input = sm.add_constant(input)

res = sm.OLS(output, input).fit() 
res.summary() 
print 'MSE:',locale.format("%d",metrics.mean_squared_error(res.predict(input), output),grouping=True)


pd.get_dummies(train_X['teamID'])

train_X = train_X.join(pd.get_dummies(train_X['teamID']))
train_X = train_X.drop('teamID',1)












