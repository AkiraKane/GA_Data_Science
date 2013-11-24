# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 16:38:38 2013

@author: alexandersedgwick
"""

import csv
import numpy as np
import pandas as pd
from pandas import DataFrame, concat
import scipy.stats as stats
from sklearn import cross_validation, linear_model, metrics, decomposition, cluster
from sklearn.preprocessing import scale
import pylab as pl

from dateutil import parser
import statsmodels.api as sm
import matplotlib.pyplot as plt
import random


#from numpy import inf

#Load the data
b2011 = pd.read_csv('C:/baseball/baseball_training_2011.csv')
b2011 = b2011.drop(['bats','throws','playerID', 'lahmanID', 'managerID','birthDate','birthMonth','birthDay',
'birthCountry', 'birthState', 'birthCity', 'deathYear', 'deathMonth', 'deathDay', 'deathCountry',
'deathState', 'deathCity', 'nameFirst','nameLast', 'nameNote', 'nameGiven', 'nameNick', 'debut', 'finalGame', 'college','lahman40ID', 'lahman45ID', 'retroID', 'holtzID',
'bbrefID', 'deathDate','G_old', 'hofID','stint'], axis =1)
b2011['Age'] = b2011['yearID'] - b2011['birthYear']
b2011 = b2011.drop(['yearID','birthYear'],axis=1)


#Explore the data and correlations
#pd.tools.plotting.scatter_matrix(b2011.drop(['teamID','lgID'],axis=1), alpha=0.2, diagonal = 'hist')


#Create new variables based on sabermetrics
b2011['AB_per_G'] = b2011.AB/b2011.G
b2011['BAVG'] = b2011.H/b2011.AB
b2011['OB_Per'] =(b2011.H + b2011.BB + b2011.HBP)/(b2011.AB + b2011.BB + b2011.HBP + b2011.SF)
b2011['Tot_Base'] = (b2011.H + (2 * b2011.X2B) + (3 * b2011.X3B) + (4 * b2011.HR))
b2011['Slug_Avg'] = b2011.Tot_Base/b2011.AB
b2011['Tot_Avg'] = ((b2011.Tot_Base + b2011.BB + b2011.HBP + b2011.SB - b2011.CS)/(b2011.AB - b2011.H + b2011.CS + b2011.GIDP))
b2011['Time_on_Base'] = (b2011.H + b2011.BB + b2011.HBP)
b2011 = b2011.replace([inf, -inf], np.nan)
b2011 = b2011.dropna()
slimdata = b2011


#Add Team Groupings
#grouped = slimdata.groupby('teamID')
#salary_list = grouped['salary'].aggregate(np.sum)
#salary_list.sort(1)


#Split into 6 groups
slimdata['group_A'] = 0
slimdata['group_B'] = 0
slimdata['group_C'] = 0
slimdata['group_D'] = 0
slimdata['group_E'] = 0
slimdata['group_A'][slimdata['teamID'].isin(['NYA','PHI','BOS','CHA','CHN','SFN'])] = 1
slimdata['group_B'][slimdata['teamID'].isin(['LAA','NYN','MIN','DET','LAN','SLN'])] = 1
slimdata['group_C'][slimdata['teamID'].isin(['COL','ATL','MIL','TEX','SEA','BAL'])] = 1
slimdata['group_D'][slimdata['teamID'].isin(['HOU','WAS','CIN','FLO','ARI','OAK'])] = 1
slimdata['group_E'][slimdata['teamID'].isin(['CLE','TOR','PIT','TBA','SDN','KCA'])] = 1
slimdata = slimdata.drop(['teamID',],axis=1)

slimdata = slimdata.join(pd.get_dummies(slimdata['lgID']))
slimdata = slimdata.drop(['lgID'],axis=1)
slimdata = slimdata.replace([inf, -inf], np.nan)
slimdata = slimdata.dropna()


#Drop at least one of the dummies
slimdata = slimdata.drop(['NL','group_E'], axis=1)
X = slimdata.drop(['salary'], axis=1)
y = slimdata['salary']

#lm = linear_model.LinearRegression() # 1.41752059101e+13
#lm = linear_model.Ridge() # 1.42704917393e+13
#lm = linear_model.ElasticNet() #1.54627795848e+13
#lm = linear_model.LassoLars(alpha=.1) #1.48868703505e+13
#lm = linear_model.BayesianRidge() #1.57922421351e+13


lm = linear_model.Lasso() # 1.41752059694e+13
lm.fit(X, y)
#
# Checking performance, roughly .19
print 'R-Squared:',lm.score(X, y)
# Checking MSE, roughly terrible
print 'MSE:',metrics.mean_squared_error(lm.predict(X), y)


#############################################################################
# Model 2 - Identify Pitchers/Hitters
#############################################################################


random.seed(1)
cls = cluster.k_means(slimdata[['G','BAVG']],2)
slimdata['Hitter_Flag'] = c=list(cls[1])
hitters = slimdata[slimdata['Hitter_Flag'] == 0]
pitchers = slimdata[slimdata['Hitter_Flag'] == 1]
X = slimdata.drop(['salary'], axis=1)
y = slimdata['salary']

#Verify groupings with plots
plt.scatter(pitchers.G, pitchers.salary, cmap=plt.cm.jet, c='blue')
plt.show()
plt.scatter(hitters.G, hitters.salary, cmap=plt.cm.jet, c='red')
plt.show()


#Regressions
X_P = pitchers.drop(['salary'], axis=1)
y_P = pitchers['salary']
X_H = hitters.drop(['salary'], axis=1)
y_H = hitters['salary']

#Hitters
lm = linear_model.Ridge()
lm = linear_model.LinearRegression()
lm.fit(X_H, y_H)

#
# Checking performance, roughly .19
print 'R-Squared:',lm.score(X_H, y_H)
# Checking MSE, roughly terrible
print 'MSE:',metrics.mean_squared_error(lm.predict(X_H), y_H)



#Pitchers
lm = linear_model.Ridge()
lm = linear_model.LinearRegression()
lm.fit(X_P, y_P)
#
# Checking performance, roughly .19
print 'R-Squared:',lm.score(X_P, y_P)
# Checking MSE, roughly terrible
print 'MSE:',metrics.mean_squared_error(lm.predict(X_P), y_P)



#############################################################################
# Prep Test Data and run regressions
#############################################################################


b2012 = pd.read_csv('C:/baseball/baseball_test_2012.csv')
b2012 = b2012.drop(['lahmanID', 'birthDate','birthMonth','birthDay',
'birthCountry', 'birthCity', 'nameFirst','nameLast', 'stint','bats','throws'], axis =1)
b2012['Age'] = b2012['yearID'] - b2012['birthYear']
b2012 = b2012.drop(['birthYear'],axis=1)
b2012['AB_per_G'] = b2012.AB/b2012.G
b2012['BAVG'] = b2012.H/b2012.AB
b2012['OB_Per'] =(b2012.H + b2012.BB + b2012.HBP)/(b2012.AB + b2012.BB + b2012.HBP + b2012.SF)
b2012['Tot_Base'] = (b2012.H + (2 * b2012.X2B) + (3 * b2012.X3B) + (4 * b2012.HR))
b2012['Slug_Avg'] = b2012.Tot_Base/b2012.AB
b2012['Tot_Avg'] = ((b2012.Tot_Base + b2012.BB + b2012.HBP + b2012.SB - b2012.CS)/(b2012.AB - b2012.H + b2012.CS + b2012.GIDP))
b2012['Time_on_Base'] = (b2012.H + b2012.BB + b2012.HBP)
b2012 = b2012.replace([inf, -inf], np.nan)
b2012 = b2012.dropna()
slimdata12 = b2012


#Split into 6 groups
slimdata12['group_A'] = 0
slimdata12['group_B'] = 0
slimdata12['group_C'] = 0
slimdata12['group_D'] = 0
slimdata12['group_E'] = 0
slimdata12['group_A'][slimdata12['teamID'].isin(['NYA','PHI','BOS','CHA','CHN','SFN'])] = 1
slimdata12['group_B'][slimdata12['teamID'].isin(['LAA','NYN','MIN','DET','LAN','SLN'])] = 1
slimdata12['group_C'][slimdata12['teamID'].isin(['COL','ATL','MIL','TEX','SEA','BAL'])] = 1
slimdata12['group_D'][slimdata12['teamID'].isin(['HOU','WAS','CIN','FLO','ARI','OAK'])] = 1
slimdata12['group_E'][slimdata12['teamID'].isin(['CLE','TOR','PIT','TBA','SDN','KCA'])] = 1
slimdata12 = slimdata12.drop(['teamID',],axis=1)
slimdata12 = slimdata12.join(pd.get_dummies(slimdata12['lgID']))
slimdata12 = slimdata12.drop(['lgID'],axis=1)
slimdata12 = slimdata12.replace([inf, -inf], np.nan)
slimdata12 = slimdata12.dropna()
slimdata12 = slimdata12.drop(['NL','group_E'], axis=1)



### ID Pitchers
cls = cluster.k_means(slimdata12[['G','AL']],2)
slimdata12['Hitter_Flag'] = c=list(cls[1])
pitchers12 = slimdata12[slimdata12['Hitter_Flag'] == 0]
hitters12 = slimdata12[slimdata12['Hitter_Flag'] == 1]
X = slimdata12.drop(['salary'], axis=1)
y = slimdata12['salary']
plt.scatter(pitchers12.G, pitchers12.salary, cmap=plt.cm.jet, c='blue')
plt.show()
plt.scatter(hitters12.G, hitters12.salary, cmap=plt.cm.jet, c='red')
plt.show()



# Run Models
#Regressions
X_P12 = pitchers12.drop(['salary'], axis=1)
y_P12 = pitchers12['salary']
X_H12 = hitters12.drop(['salary'], axis=1)
y_H12 = hitters12['salary']

#Hitters
lm = linear_model.Ridge()
lm = linear_model.LinearRegression()
lm.fit(X_H12.drop(['playerID','yearID'],axis=1), y_H12)


#
# Checking performance, roughly .19
print 'R-Squared:',lm.score(X_H12.drop(['playerID','yearID'],axis=1), y_H12)
# Checking MSE, roughly terrible
print 'MSE:',metrics.mean_squared_error(lm.predict(X_H12.drop(['playerID','yearID'],axis=1)), y_H12)
hitters12['predicted'] = lm.predict(X_H12.drop(['playerID','yearID'],axis=1))



#Pitchers
lm = linear_model.Ridge()
lm = linear_model.LinearRegression()
lm.fit(X_P12.drop(['playerID','yearID'],axis=1), y_P12)
#
# Checking performance, roughly .19
print 'R-Squared:',lm.score(X_P12.drop(['playerID','yearID'],axis=1), y_P12)
# Checking MSE, roughly terrible
print 'MSE:',metrics.mean_squared_error(lm.predict(X_P12.drop(['playerID','yearID'],axis=1)), y_P12)


#Prep Submission File
pitchers12['predicted'] = lm.predict(X_P12.drop(['playerID','yearID'],axis=1))
submission = hitters12.append(pitchers12)
b2012_csv = submission[['playerID','yearID', 'salary','predicted']]
b2012_csv.to_csv('C:/submission.csv')


#############################################################################
# Sandbox
#############################################################################
x_train, x_test, y_train, y_test = cross_validation.train_test_split(X,y, test_size=.3)
#PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=1)
pca.fit(X)
print(pca.explained_variance_ratio_)
X_PCA = pca.transform(X)
model_pca = sm.OLS(y, X_PCA)
ols_results_pca = model_pca.fit()
print ols_results_pca.summary()
from sklearn.metrics import mean_squared_error
mean_squared_error(y, ols_results_pca.predict(X_PCA))
pca.fit(X)
pl.figure(1, figsize=(4, 3))
pl.clf()
pl.axes([.2, .2, .7, .7])
pl.plot(pca.explained_variance_, linewidth=2)
pl.axis('tight')
pl.xlabel('n_components')
pl.ylabel('explained_variance_')
#Feature Selection using recursive feature selection
from sklearn.feature_selection import *
from sklearn import linear_model
estimator = linear_model.LinearRegression()
selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(X, y)
selector.support_
selector.ranking_
X_RFECV = X[X.columns[selector.support_]]
#Feature Selection based on Tree Classifier
from sklearn.ensemble import ExtraTreesClassifier
selector = ExtraTreesClassifier(criterion = 'gini', compute_importances=True)
selector = selector.fit(X, y)
X_Tree = selector.transform(X)
#Initial Test Regressions
X = X.drop(['H','X2B','X3B','HR','SB','IBB','SH','SF','AB_per_G','Tot_Base','BAVG','Time_on_Base'],axis=1)
X = X.drop(['R','HBP','GIDP','G_batting'],axis=1)
X = X.drop(['CS','group_D'],axis=1)
model1 = sm.OLS(y, X)
ols_results1 = model1.fit()
print ols_results1.summary()
from sklearn.metrics import mean_squared_error
mean_squared_error(y, ols_results1.predict(X))
13756411927357
model2 = sm.OLS(y, X_RFECV)
ols_results2 = model2.fit()
print ols_results2.summary()
mean_squared_error(y, ols_results2.predict(X_RFECV))
14173528459103.535
model3 = sm.OLS(y, X_Tree)
ols_results3 = model3.fit()
print ols_results3.summary()
mean_squared_error(y, ols_results3.predict(X_Tree))
16246800373211.918
# Lasso Model
clf = linear_model.Lasso(alpha = 0.1)
clf.fit(X,y)
#clf.predict(X)
mean_squared_error(y, clf.predict(X))
#l1 based feature selection
from sklearn.svm import LinearSVC
X.shape
X_new = LinearSVC(C=0.01, penalty="l1", dual=False).fit_transform(X, y)
X_new.shape
model3 = sm.OLS(y, X_new)
ols_results3 = model3.fit()
print ols_results3.summary()
mean_squared_error(y, ols_results3.predict(X_new))
#SVR
from sklearn import svm
clf = svm.SVR()
model = clf.fit(X, y)
mean_squared_error(y, model.predict(X))
#Identify Pitchers vs. Hitters using Kmeans
from sklearn import cluster
from numpy import random
from pandas import DataFrame, concat
from matplotlib import pyplot as plt
random.seed(1)
cls = cluster.k_means(data_norm[['G','AB_per_G']],2)
data_norm['Hitter_Flag'] = c=list(cls[1])
hitters = data_norm[data_norm['Hitter_Flag'] == 0]
pitchers = data_norm[data_norm['Hitter_Flag'] == 1]
plt.scatter(pitchers.AB_per_G, pitchers.salary, cmap=plt.cm.jet, c='blue')
plt.show()
#Regression for Pitchers
X = pitchers[['AB_per_G','Age']]
y = pitchers['salary']
model1 = sm.OLS(y, X)
ols_results1 = model1.fit()
print ols_results1.summary()
#Regression for Hitters
X = hitters[['height','weight','Tot_Base','Age']]
y = hitters['salary']
model2 = sm.OLS(y, X)
ols_results2 = model2.fit()
print ols_results2.summary()