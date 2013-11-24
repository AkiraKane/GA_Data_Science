# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 18:36:40 2013

@author: alexandersedgwick
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


data = pd.read_csv("/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/baseball/baseball.csv")

#slim the data down

slimdata = data.drop(['playerID', 'lahmanID', 'managerID', 'birthYear', 'birthMonth', 'birthDay', 
'birthCountry', 'birthState', 'birthCity', 'deathYear', 'deathMonth', 'deathDay', 'deathCountry', 
'deathState', 'deathCity', 'nameFirst','nameLast', 'nameNote', 'nameGiven', 'nameNick','bats', 
'throws', 'debut', 'finalGame', 'college','lahman40ID', 'lahman45ID', 'retroID', 'holtzID', 
'bbrefID', 'deathDate', 'birthDate','teamID', 'lgID', 'stint','G_batting','X2B', 'X3B',
'CS', 'SO', 'IBB', 'HBP', 'SH', 'SF', 'GIDP', 'G_old', 'hofID', 'yearID'], axis =1)

slimdata = slimdata.dropna()


slimdata['random'] = np.random.randn(len(slimdata))
slimdata = slimdata[slimdata.random > 1]
slimdata.AB = np.log(slimdata.AB)
slimdata = slimdata.replace([inf, -inf], np.nan)
slimdata = slimdata.dropna()
# We could go in and do this to all variables that are log skewed


#histogram
pd.tools.plotting.scatter_matrix(slimdata, alpha=0.2, diagonal='hist')
plt.show()


#kernel Density
pd.tools.plotting.scatter_matrix(slimdata, alpha=0.2, diagonal='kde')
plt.show()

#Defining IV and DV
X = np.array(slimdata.drop(['salary'], axis = 1))
y = np.array(slimdata['salary'])


#Regression model

model = sm.OLS(y, X)
ols_results = model.fit()
print ols_results.summary()



#Box plot for outliers
slimdata.boxplot()
plt.show()
slimdata.drop(['salary'], axis = 1).boxplot()
plt.show()


#Data Scaling
data_norm = pd.DataFrame(scale(slimdata), index=slimdata.index, columns=slimdata.columns)
data_norm.boxplot()
plt.show()



# Re-Run regression model
X = np.array(data_norm.drop(['salary'], axis = 1))
y = np.array(data_norm['salary'])

model2 = sm.OLS(y, X)
ols_results2 = model2.fit()
print ols_results2.summary()


#Influence Plot

sm.graphics.influence_plot(results2)


influence = results2.get_influence()
(d,p) = influence.cooks_distance
plt.stem(np.arange(len(d)),d,markerfmt=",")




#Residuals vs. Fitted Plot
plt.scatter(results.norm_resid(), results.fittedvalues)
plt.xlabel('Fitted Values')
plt.ylabel('Normalized residuals')


#QQ Plot
stats.probplot(slimdata.AB, dist="norm", plot=pylab)
pylab.show()
stats.probplot(data_norm['weight'], dist="norm", plot=pylab)
pylab.show()


res_dropped = ols_results.params / ols_results2.params * 100 - 100


################################################################################################
#### Start at the beginning and look at different values - combine where possible.
################################################################################################

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


data = pd.read_csv("/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/baseball/baseball.csv")

# playerID       Player ID code
# yearID         Year
# stint          player's stint (order of appearances within a season)
# teamID         Team
# lgID           League
# G              Games
# G_batting      Game as batter
# AB             At Bats
# R              Runs Scored
# H              Hits
# 2B             Doubles
# 3B             Triples
# HR             Homeruns - Ind
# RBI            Runs Batted In
# SB             Stolen Bases
# CS             Caught Stealing
# BB             Base on Balls
# SO             Strikeouts
# IBB            Intentional walks
# HBP            Hit by pitch
# SH             Sacrifice hits
# SF             Sacrifice flies
# GIDP           Grounded into double plays
# G_Old          Old version of games (deprecated)



data['AB_per_G'] = data.AB/data.G
data['BAVG'] = data.H/data.AB
data['STBS_Per'] =data.SB/(data.SB+data.CS)
data['H_per_R'] = data.H/data.R
data['W_to_SO'] =data.BB/data.SO
data['OB_Per'] =(data.H + data.BB + data.HBP)/(data.AB + data.BB + data.HBP + data.SF)
data['Tot_Base'] =  (data.H + (2 * data.X2B) + (3 * data.X3B) + (4 * data.HR))
data['Slug_Avg'] = data.Tot_Base/data.AB
data['Tot_Avg'] = ((data.Tot_Base + data.BB + data.HBP + data.SB - data.CS)/(data.AB - data.H + data.CS + data.GIDP))
data['Time_on_Base'] = (data.H + data.BB + data.HBP)


slimdata = data.drop(['playerID', 'lahmanID', 'managerID','birthYear','birthMonth','birthDay',
'birthCountry', 'birthState', 'birthCity', 'deathYear', 'deathMonth', 'deathDay', 'deathCountry', 
'deathState', 'deathCity', 'nameFirst','nameLast', 'nameNote', 'nameGiven', 'nameNick', 'debut', 'finalGame', 'college','lahman40ID', 'lahman45ID', 'retroID', 'holtzID', 
'bbrefID', 'deathDate','G_old', 'hofID', 'yearID','birthDate','teamID','lgID','stint','G','G_batting',
'AB','R','H','X2B','X3B','HR','RBI','SB','CS','BB','SO','IBB','HBP','SH','SF','GIDP'], axis =1)

slimdata = slimdata.dropna()

#histogram
pd.tools.plotting.scatter_matrix(slimdata, alpha=0.2, diagonal='hist')
plt.show()



stats.pearsonr(slimdata.AB, slimdata.G) #AB per game
stats.pearsonr(slimdata.H, slimdata.AB) # Batting Avg H/AB
stats.pearsonr(slimdata.CS, slimdata.SB) #SB/(CS+SB) stolen base percentage
stats.pearsonr(slimdata.H, slimdata.R) #Hits per run
stats.pearsonr(slimdata.BB, slimdata.SO) #Walk to strikeout ratio

# AB/HR at bats per home run
# on base percentage  (H + BB + HBP)/(AB + BB + HBP + SF)
# Total Bases (TB)  [H + 2B + (2 * 3B) + (3 * HR)] or [1B + (2 * 2B) + (3 * 3B) + (4 * HR)]
# Slugging average (TB/AB)
# Total Average  [(TB + BB + HBP + SB - CS)/(AB - H + CS + GIDP)]
# Times on Base (H + BB + HBP)


# VORP - Outs Outs are calculated by simply taking at-bats and subtracting hits then adding in various outs that don't count toward at-bats: sacrifice hits, sacrifice flies, caught stealing, and grounded into double-play
#values of runs created


#Data Scaling
data_norm = pd.DataFrame(scale(slimdata), index=slimdata.index, columns=slimdata.columns)
data_norm.boxplot()
plt.show()



# Re-Run regression model
X = np.array(data_norm.drop(['salary'], axis = 1))

X = np.array(data_norm['Tot_Base'], axis = 1)
y = np.array(data_norm['salary'])

model3 = sm.OLS(y, X)
ols_results3 = model3.fit()
print ols_results3.summary()


