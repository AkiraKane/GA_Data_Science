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
b2011 = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/baseball/baseball_training_2011.csv')
b2011 = b2011.drop(['bats','throws','G_batting','playerID', 'lahmanID', 'managerID','birthDate','birthMonth','birthDay',
'birthCountry', 'birthState', 'birthCity', 'deathYear', 'deathMonth', 'deathDay', 'deathCountry',
'deathState', 'deathCity', 'nameFirst','nameLast', 'nameNote', 'nameGiven', 'nameNick', 'debut', 'finalGame', 'college','lahman40ID', 'lahman45ID', 'retroID', 'holtzID',
'bbrefID', 'deathDate','G_old', 'hofID','stint'], axis =1)
b2011['Age'] = b2011['yearID'] - b2011['birthYear']
b2011 = b2011.drop(['yearID','birthYear'],axis=1)



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

#plt.hist(log(y))
#plt.show

#plt.hist(log(slimdata['salary'][slimdata['salary']>5000000]))
#plt.show

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
lm_H = linear_model.Ridge()
lm_H.fit(X_H, y_H)

#
# Checking performance, roughly .19
print 'R-Squared:',lm_H.score(X_H, y_H)
# Checking MSE, roughly terrible
print 'MSE:',metrics.mean_squared_error(lm_H.predict(X_H), y_H)



#Pitchers
lm_P = linear_model.Ridge()
lm_P.fit(X_P, y_P)
#
# Checking performance, roughly .19
print 'R-Squared:',lm_P.score(X_P, y_P)
# Checking MSE, roughly terrible
print 'MSE:',metrics.mean_squared_error(lm_P.predict(X_P), y_P)



#############################################################################
# Prep Test Data and run regressions
#############################################################################


b2012 = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/baseball/baseball_test_2012.csv')
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
cls = cluster.k_means(slimdata12[['G','BAVG']],2)
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

##Hitters
#lm = linear_model.Ridge()
#lm = linear_model.LinearRegression()
#lm.fit(X_H12.drop(['playerID','yearID'],axis=1), y_H12)

lm_H = linear_model.Ridge()
lm_H.fit(X_H, y_H)



#
# Checking performance, roughly .19
print 'R-Squared:',lm_H.score(X_H12.drop(['playerID','yearID'],axis=1), y_H12)
# Checking MSE, roughly terrible
print 'MSE:',metrics.mean_squared_error(lm_H.predict(X_H12.drop(['playerID','yearID'],axis=1)), y_H12)
hitters12['predicted'] = lm_H.predict(X_H12.drop(['playerID','yearID'],axis=1))



#Pitchers
#lm = linear_model.Ridge()
#lm = linear_model.LinearRegression()
#lm.fit(X_P12.drop(['playerID','yearID'],axis=1), y_P12)
#
lm_P = linear_model.Ridge()
lm_P.fit(X_P, y_P)


# Checking performance, roughly .19
print 'R-Squared:',lm_P.score(X_P12.drop(['playerID','yearID'],axis=1), y_P12)
# Checking MSE, roughly terrible
print 'MSE:',metrics.mean_squared_error(lm_P.predict(X_P12.drop(['playerID','yearID'],axis=1)), y_P12)



#Prep Submission File
pitchers12['predicted'] = lm_P.predict(X_P12.drop(['playerID','yearID'],axis=1))
submission = hitters12.append(pitchers12)
b2012_csv = submission[['playerID','yearID', 'salary','predicted']]

((submission['salary'] - submission['predicted'])**2).mean()

b2012_csv.to_csv('/users/alexandersedgwick/dropbox/development/ga/ongoing projects/baseball/submission.csv')
