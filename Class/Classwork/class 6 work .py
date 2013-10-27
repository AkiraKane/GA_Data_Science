# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 19:43:01 2013

@author: alexandersedgwick
"""

#%% Predicting stopping distance by speed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import *

#import data
stopping = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/data/stopping_dist.csv')
mpg = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/data/mpg_data.csv')



plt.scatter(stopping.speed, stopping.dist)


stop_lm = linear_model.LinearRegression()
stop_lm.fit(stopping[['speed']], stopping[['dist']])

stop_lm.coef_
stop_lm.intercept_

# Display our SSE:
mean((stop_lm.predict(stopping[['speed']]) - stopping[['dist']].values) ** 2)
# Scoring our model (closer to 1 is better!)
regr.score(stopping[['speed']], stopping[['dist']])


plt.scatter(stopping[['speed']], stopping[['dist']])
plt.plot(stopping[['speed']], regr.predict(stopping[['speed']]), color='blue', linewidth=3)
plt.show()


#%% Multivariate regression of mpg 

output = mpg['MPG.city']
#input =mpg[['Min.Price','Price','Max.Price','MPG.city','Cylinders','EngineSize','Horsepower','RPM','Rev.per.mile','Fuel.tank.capacity','Passengers','Length','Wheelbase','Width','Turn.circle','Rear.seat.room','Luggage.room','Weight']]
mpg.dtypes

input =mpg[['EngineSize','Horsepower','RPM','Rev.per.mile','Fuel.tank.capacity','Length','Width','Weight']]



city_lm = linear_model.LinearRegression()
city_lm.fit(input,output)

city_lm.coef_
city_lm.intercept_
city_lm.score(input,output)


def f_regression(X,Y):
   import sklearn
   return sklearn.feature_selection.f_regression(X,Y,center=False) #center=True (the default) would not work ("ValueError: center=True only allowed for dense data") but should presumably work in general
from sklearn.feature_selection import SelectKBest


featureSelector = SelectKBest(score_func=f_regression,k=2)
test = featureSelector.fit(input,output)

print [1+zero_based_index for zero_based_index in list(featureSelector.get_support(indices=True))]




scatter_matrix(mpg[['EngineSize','Horsepower','RPM','Rev.per.mile','Fuel.tank.capacity','Length','Width','Weight','MPG.city']], alpha=.2, figsize=(6,6), diagonal='kds')

input =mpg[['RPM','Rev.per.mile']]
