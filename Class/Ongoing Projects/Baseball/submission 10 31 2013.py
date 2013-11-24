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

#Load the data
b2011 = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/baseball/baseball_training_2011.csv')

b2011['Age'] = b2011['yearID'] - b2011['birthYear']
b2011 = b2011.drop(['bats','throws','G_batting', 'lahmanID', 'managerID','birthDate','birthMonth','birthDay',
'birthCountry', 'birthState', 'birthCity', 'deathYear', 'deathMonth', 'deathDay', 'deathCountry',
'deathState', 'deathCity', 'nameFirst','nameLast', 'nameNote', 'nameGiven', 'nameNick', 'debut', 'finalGame', 'college','lahman40ID', 'lahman45ID', 'retroID', 'holtzID',
'bbrefID', 'deathDate','G_old', 'hofID','stint', 'yearID','birthYear'], axis =1)

b2012 = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/baseball/baseball_test_2012.csv')
b2012 = b2012[['playerID','salary']]
b2011 = pd.merge(b2012, b2011, on='playerID', how='outer', suffixes=('_2012', '_2011'))

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

#############################################################################
# Identify Pitchers/Hitters
#############################################################################


random.seed(1)
cls = cluster.k_means(slimdata[['G','BAVG']],2)
slimdata['Hitter_Flag'] = c=list(cls[1])
pitchers = slimdata[slimdata['Hitter_Flag'] == 0]
hitters = slimdata[slimdata['Hitter_Flag'] == 1]
X = slimdata.drop(['salary_2012'], axis=1)
y = slimdata['salary_2012']

#Verify groupings with plots
plt.scatter(pitchers.G, pitchers.salary_2012, cmap=plt.cm.jet, c='blue')
plt.show()
plt.scatter(hitters.G, hitters.salary_2012, cmap=plt.cm.jet, c='red')
plt.show()

#Regression Inputs for Each Group
X_P = pitchers.drop(['salary_2012'], axis=1)
y_P = pitchers['salary_2012']
X_H = hitters.drop(['salary_2012'], axis=1)
y_H = hitters['salary_2012']


#Average salaries for each group
p_min = y_P.min()
h_min = y_H.min()


##Hitters Regression
lm_H = linear_model.Ridge()
lm_H.fit(X_H.drop(['playerID'],axis=1), y_H)
hitters['predicted'] = lm_H.predict(X_H.drop(['playerID'],axis=1))
hitters['predicted'][hitters['predicted'] < h_min] = h_min
# Checking MSE
print 'MSE for Hitters:',metrics.mean_squared_error(hitters['predicted'], y_H)

#Pitchers Regression
lm_P = linear_model.Ridge()
lm_P.fit(X_P.drop(['playerID'],axis=1), y_P)
pitchers['predicted'] = lm_P.predict(X_P.drop(['playerID'],axis=1))
pitchers['predicted'][pitchers['predicted'] < p_min] = h_min

# Checking MSE, roughly terrible
print 'MSE for Pitchers:',metrics.mean_squared_error(pitchers['predicted'], y_P)

#Prep Submission File
submission = hitters.append(pitchers)
b2012_csv = submission[['playerID', 'salary_2012','predicted']]
print 'Combined MSE:', ((submission['salary_2012'] - submission['predicted'])**2).mean()
b2012_csv.to_csv('/users/alexandersedgwick/dropbox/development/ga/ongoing projects/baseball/submission.csv')
