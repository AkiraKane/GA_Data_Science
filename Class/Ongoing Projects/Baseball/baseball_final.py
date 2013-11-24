# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 10:36:42 2013
@author: asedgwick
"""
import csv
import numpy as np
import pandas as pd
from dateutil import parser
import pylab as pl
import statsmodels.api as sm
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import scale
from numpy import inf
import scipy.stats as stats
import pylab
from sklearn import metrics
#Load the data
b2011 = pd.read_csv('C:/baseball/baseball_training_2011.csv')
b2012 = pd.read_csv('C:/baseball/baseball_test_2012.csv')
b2011['AB_per_G'] = b2011.AB/b2011.G
b2011['BAVG'] = b2011.H/b2011.AB
b2011['OB_Per'] =(b2011.H + b2011.BB + b2011.HBP)/(b2011.AB + b2011.BB + b2011.HBP + b2011.SF)
b2011['Tot_Base'] = (b2011.H + (2 * b2011.X2B) + (3 * b2011.X3B) + (4 * b2011.HR))
b2011['Slug_Avg'] = b2011.Tot_Base/b2011.AB
b2011['Tot_Avg'] = ((b2011.Tot_Base + b2011.BB + b2011.HBP + b2011.SB - b2011.CS)/(b2011.AB - b2011.H + b2011.CS + b2011.GIDP))
b2011['Time_on_Base'] = (b2011.H + b2011.BB + b2011.HBP)
slimdata = b2011.drop(['bats','throws', 'lahmanID', 'managerID','birthDate','birthMonth','birthDay',
'birthCountry', 'birthState', 'birthCity', 'deathYear', 'deathMonth', 'deathDay', 'deathCountry',
'deathState', 'deathCity', 'nameFirst','nameLast', 'nameNote', 'nameGiven', 'nameNick', 'debut', 'finalGame', 'college','lahman40ID', 'lahman45ID', 'retroID', 'holtzID',
'bbrefID', 'deathDate','G_old', 'hofID','stint','lgID'], axis =1)
slimdata['Age'] = slimdata['yearID'] - slimdata['birthYear']
slimdata = slimdata.drop(['yearID','birthYear'],axis=1)
slimdata = slimdata.replace([inf, -inf], np.nan)
slimdata = slimdata.dropna()
#Add Team Groupings
grouped = slimdata.groupby('teamID')
salary_list = grouped['salary'].aggregate(np.sum)
salary_list.sort(1)
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
data_norm = slimdata.drop(['playerID','teamID','group_A','group_B','group_C','group_D','group_E'],axis=1)
#Data Scaling
data_norm = pd.DataFrame(scale(data_norm), index=data_norm.index, columns=data_norm.columns)
data_norm = data_norm.replace([inf, -inf], np.nan)
data_norm['playerID'] = slimdata['playerID']
data_norm['group_A'] = slimdata['group_A']
data_norm['group_B'] = slimdata['group_B']
data_norm['group_C'] = slimdata['group_C']
data_norm['group_D'] = slimdata['group_D']
data_norm['group_E'] = slimdata['group_E']
data_norm = data_norm.dropna()
#Identify Pitchers vs. Hitters using Kmeans
from sklearn import cluster
from numpy import random
from pandas import DataFrame, concat
from matplotlib import pyplot as plt
random.seed(1)
cls = cluster.k_means(data_norm[['G','AB_per_G']],2)
data_norm['Hitter_Flag'] = c=list(cls[1])
pitchers = data_norm[data_norm['Hitter_Flag'] == 0]
hitters = data_norm[data_norm['Hitter_Flag'] == 1]
plt.scatter(pitchers.G, pitchers.salary, cmap=plt.cm.jet, c='blue')
plt.show()
plt.scatter(hitters.G, hitters.salary, cmap=plt.cm.jet, c='red')
plt.show()
#Regression for Pitchers
X = pitchers[['AB_per_G','Age','group_A','group_B','group_C','group_D']]
y = pitchers['salary']
model1 = sm.OLS(y, X)
ols_results1 = model1.fit()
print ols_results1.summary()
#Regression for Hitters
X = hitters[['height','weight','Tot_Base','Age','group_A','group_B','group_C','group_D']]
y = hitters['salary']
model2 = sm.OLS(y, X)
ols_results2 = model2.fit()
print ols_results2.summary()
#Predict Final Predictions for 2012
b2012['AB_per_G'] = b2012.AB/b2012.G
b2012['BAVG'] = b2012.H/b2012.AB
b2012['OB_Per'] =(b2012.H + b2012.BB + b2012.HBP)/(b2012.AB + b2012.BB + b2012.HBP + b2012.SF)
b2012['Tot_Base'] = (b2012.H + (2 * b2012.X2B) + (3 * b2012.X3B) + (4 * b2012.HR))
b2012['Slug_Avg'] = b2012.Tot_Base/b2012.AB
b2012['Tot_Avg'] = ((b2012.Tot_Base + b2012.BB + b2012.HBP + b2012.SB - b2012.CS)/(b2012.AB - b2012.H + b2012.CS + b2012.GIDP))
b2012['Time_on_Base'] = (b2012.H + b2012.BB + b2012.HBP)
slimdata12 = b2012[['salary','playerID','yearID','birthYear','teamID','G','AB_per_G','height','weight','Tot_Base']]
slimdata12['Age'] = slimdata12['yearID'] - slimdata12['birthYear']
slimdata12 = slimdata12.drop(['birthYear'],axis=1)
slimdata12 = slimdata12.replace([inf, -inf], np.nan)
slimdata12 = slimdata12.dropna()
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
data_norm12 = slimdata12.drop(['yearID','playerID','teamID','group_A','group_B','group_C','group_D','group_E'],axis=1)
#Data Scaling
data_norm12 = pd.DataFrame(scale(data_norm12), index=data_norm12.index, columns=data_norm12.columns)
data_norm12 = data_norm12.replace([inf, -inf], np.nan)
data_norm12['group_A'] = slimdata12['group_A']
data_norm12['group_B'] = slimdata12['group_B']
data_norm12['group_C'] = slimdata12['group_C']
data_norm12['group_D'] = slimdata12['group_D']
data_norm12['group_E'] = slimdata12['group_E']
data_norm12['playerID'] = slimdata12['playerID']
data_norm12['yearID'] = slimdata12['yearID']
data_norm12 = data_norm12.dropna()
#Identify Pitchers vs. Hitters using Kmeans
from sklearn import cluster
from numpy import random
from pandas import DataFrame, concat
from matplotlib import pyplot as plt
random.seed(1)
cls = cluster.k_means(data_norm12[['G','AB_per_G']],2)
data_norm12['Hitter_Flag'] = c=list(cls[1])
pitchers12 = data_norm12[data_norm12['Hitter_Flag'] == 0]
hitters12 = data_norm12[data_norm12['Hitter_Flag'] == 1]
plt.scatter(pitchers12.AB_per_G, pitchers12.salary, cmap=plt.cm.jet, c='blue')
plt.show()
plt.scatter(hitters12.AB_per_G, hitters12.salary, cmap=plt.cm.jet, c='red')
plt.show()
#Regression for Pitchers
X = pitchers12[['AB_per_G','Age','group_A','group_B','group_C','group_D']]
pitchers12['predicted'] = ols_results1.predict(X)
#Regression for Hitters
X = hitters12[['height','weight','Tot_Base','Age','group_A','group_B','group_C','group_D']]
hitters12['predicted'] = ols_results2.predict(X)
# Outputting to a csv file
print "Outputting submission file as 'submission.csv'"
b2012_csv['predicted'] = lm.predict(test_X)
b2012_csv.to_csv('submission.csv')
pitchers12 = pitchers12[['playerID','yearID','salary','predicted']]
hitters12 = hitters12[['playerID','yearID','salary','predicted']]
output = pitchers12.append(hitters12)
output[['salary','predicted']] = exp(output[['salary','predicted']])
output.to_csv('C:/submission.csv')