# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 18:27:55 2013

@author: alexandersedgwick
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

#import data
mammals = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/data/sandbox/mammals.csv')

lm = linear_model.LinearRegression()
log_lm = linear_model.LinearRegression()

body = [ [x] for x in mammals['body'].values]
log_body = log_body = [ [x] for x in np.log(mammals['body'].values)]

brain = mammals['brain'].values
log_brain = np.log(mammals['brain'].values)

lm.fit(body, brain)
log_lm.fit(log_body, log_brain)




lm.intercept_
log_lm.intercept_

lm.coef_
log_lm.coef_

lm.predict(body)
mammals['predict'] = lm.predict(body)
log_lm.predict(log_body)
mammals['log_predict'] = np.exp(log_lm.predict(log_body))




#%% Plotting Predictions

# Sort by response:
mammals = mammals.sort('brain')
# Sort by prediction:
mammals_log_sort = mammals.sort('log_predict')

plt.scatter(mammals.body,mammals.brain)
plt.scatter(mammals_log_sort.body,mammals_log_sort.brain)

#%% Multivariate Regressions
smoking = pd.read_csv('http://www.ats.ucla.edu/stat/examples/chp/p081.txt', delimiter="\t")
input = [ [x, y, z] for x,y,z in zip(smoking['Price'].values, smoking['Income'].values, smoking['Age'].values)]

# More efficiently:
input = smoking[ ['Price', 'Income', 'Age'] ].values

#%% Polynomial Regressions


mammals['body_squared'] = mammals['body']**2
body_squared = [ [x, y] for x,y in zip(mammals['body'].values, mammals['body_squared'].values)]
# OR
body_squared = [ [x, y] for x,y in zip(mammals['body'].values, (mammals['body'].values)**2]    

ridge = linear_model.Ridge()
ridge.fit(body_squared, brain)


((ridge.coef_[1] * mammals['body'])**2) + ((ridge.coef_[0] * mammals['body'])) + ridge.intercept_

#mammals['polypredict']=ridge.predict(body_squared)
#mammals['polysort']= mammals.sort['polypredict']





#%% Classwork

from sklearn import feature_selection
feature_selection.univariate_selection.f_regression(input, response)



