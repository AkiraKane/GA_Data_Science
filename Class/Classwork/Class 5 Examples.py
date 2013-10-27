# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 18:32:04 2013

@author: alexandersedgwick
"""

from numpy import array, dot
from scipy import linalg
X = array([ [1, 1], [1, 2], [1, 3], [1, 4] ])
y = array([ [1], [2], [3], [4] ])
n = linalg.inv(dot(X.T, X))
k = dot(X.T, y)
coef_ = dot(n, k)

def regression(input, response):
    return dot(linalg.inv(dot(X.T, X)), dot(X.T, y))
    
    
# One line version    
dot(dot(linalg.inv(dot(X.T,X)),X.T),y) 
 



#%%
   
import pandas as pd
import matplotlib.pyplot as plt

mammals = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/data/sandbox/mammals.csv')
plt.scatter(mammals['body'], mammals['brain'])
plt.show()
plt.hist(mammals['body'], bins=range(0, 10000, 100))
plt.show()
plt.hist(mammals['brain'], bins=range(0, 10000, 100))
plt.show()


def regression(input, response):
	return dot(linalg.inv(dot(input.T,input)),dot(input.t,response))

#%%


from numpy import log
mammals['log_body'] = log(mammals['body'])
mammals['log_brain'] = log(mammals['brain'])
plt.scatter(mammals['log_body'], mammals['log_brain'])


from sklearn import linear_model
# Make the model object
regr = linear_model.LinearRegression()
# Fit the data
body = [[x] for x in mammals['body'].values]
brain = mammals['brain'].values
regr.fit(body, brain)


# Display the coefficients:
regr.coef_
# Display our SSE:
mean((regr.predict(body) - brain) ** 2)
# Scoring our model (closer to 1 is better!)
regr.score(body, brain)

plt.scatter(body, brain)
plt.plot(body, regr.predict(body), color='blue', linewidth=3)
plt.show()


mammals['predict'] = regr.predict(body)


#%%



#Go through the same steps, but this time generate a new model use the 
#log of brain and body, which we know generated a much better distribution 
#and cleaner set of data. Compare the results to the original model. 
#Remember that exp() can be used to "normalize" our "logged" values.
# Note: Make sure you start a new linear regression object!

mammals = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/data/sandbox/mammals.csv')


from numpy import log
mammals['log_body'] = log(mammals['body'])
mammals['log_brain'] = log(mammals['brain'])

from sklearn import linear_model
# Make the model object
regr = linear_model.LinearRegression()
# Fit the data
body = [[x] for x in mammals['log_body'].values]
brain = mammals['log_brain'].values
regr.fit(body, brain)


# Display the coefficients:
regr.coef_
# Display our SSE:
mean((regr.predict(body) - brain) ** 2)
# Scoring our model (closer to 1 is better!)
regr.score(body, brain)

plt.scatter(body, brain)
plt.plot(body, regr.predict(body), color='blue', linewidth=3)
plt.show()


# Using your aggregate data compiled from nytimes1-30.csv, write a 
# python script that determines the best model predicting CTR based off 
# of age and gender. Since gender is not actually numeric (it is binary),
# investigate ways to vectorize this feature. Clue: you may want two features
# now instead of one.

#%%
nytimes = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/data/sandbox/nytimes_clean.csv')
Age = [[x] for x in nytimes['Age'].values]
Gender = [[x] for x in nytimes['Gender'].values]

from pandas import *
nytimes = DataFrame(nytimes)

female = nytimes[nytimes['Gender']==0]
male =   nytimes[nytimes['Gender']==1]



#%% Male Model

from sklearn import linear_model
# Make the model object
male_regr = linear_model.LinearRegression()
# Fit the data
male_regr.fit(male[['Age']],male[['Ctr']])

# Display the coefficients:
male_regr.coef_
male_regr.intercept_
# Display our SSE:
mean((male_regr.predict(male[['Age']]) - male[['Ctr']].values) ** 2)
# Scoring our model (closer to 1 is better!)
male_regr.score(male[['Age']], male[['Ctr']])





#%% Female Model

from sklearn import linear_model
# Make the model object
female_regr = linear_model.LinearRegression()
# Fit the data
female_regr.fit(female[['Age']],female[['Ctr']])

# Display the coefficients:
female_regr.coef_
female_regr.intercept_
# Display our SSE:
mean((female_regr.predict(female[['Age']]) - female[['Ctr']].values) ** 2)
# Scoring our model (closer to 1 is better!)
female_regr.score(female[['Age']], female[['Ctr']])


#%% Plot Models


plt.figure()
plt.scatter(male[['Age']].values, male[['Ctr']].values, color='blue')
plt.plot(male[['Age']], male_regr.predict(male[['Age']]), color='blue', linewidth=3)
plt.show()

plt.scatter(female[['Age']].values, female[['Ctr']].values, color='red')
plt.plot(female[['Age']], female_regr.predict(female[['Age']]), color='red', linewidth=3)
plt.show()


#%% Comingled Model
nytimes['Male']= 0
nytimes['Female'] = 0

nytimes['Male'][nytimes['Gender']==1] = 1
nytimes['Female'][nytimes['Gender']==0] = 1



from sklearn import linear_model
# Make the model object
nyt_regr = linear_model.LinearRegression()
# Fit the data
nyt_regr.fit(nytimes[['Age','Female','Male']],nytimes[['Ctr']])

# Display the coefficients:
nyt_regr.coef_
nyt_regr.intercept_
# Display our SSE:
mean((nyt_regr.predict(nytimes[['Age','Female','Male']]) - nytimes[['Ctr']].values) ** 2)
# Scoring our model (closer to 1 is better!)
nyt_regr.score(nytimes[['Age','Female','Male']], nytimes[['Ctr']])


#%%


#Next steps


# I tried out three models in class, one was a co-mingled model with dummy variables for 
# gender as well as age. Then I built two seperate models, one for each gender.  I noticed 
# that the for the comingled model we had a very low R-squared value, we only explained 
# about 9.8% of the total variance in the click through rate with our input variables.  
# On top of that the Gender variable wasn't significant in the regression model. I 
# would characterize it as relatively poor.

# The male only model faired better with significant variables and an R-squared 
# value of 15.8%. Unfortunately the female model had an R-squared value of 4.4% though 
# with significant independent variables.

# So what are we missing?  First, I noticed that the shape of the data 
# looks almost stepwise with an elevated click through rate up to Age 20, then
# a decreased click through rate from 20-55.  From 55-65 the click through rate 
# increases and then from about 65 years old to 100, we high click through rates 
# consistent with what we say up to age 20.

# So whats going on there?  What is interesting is that the buckets we just 
# described 0-20, 20-55, 55-65 and 65+ correspond very closely to periods in one's 
# work life:
# - 0-20: Adolesnce to adulthood, no full time work
# - 20-55: Height of career and work engagement
# - 55-65: Transitional period of working
# - 65+: Retirement

# More importantly the pattern of click throughs mirror what one would expect from people 
# in each of these age buckets. Before graduating college people frequently shop and have active 
# online lives. During 20-55/65 people spend 40 hours or more of their week at work and likely 
# are using the web for work functions (research etc.) and far less likely to casually engage advertisements.  
# Finally during the 55/65+ period people again have the free time to shop online and 
# are more likely to be using the internet for personal business rather than work.

# It is also interesting that in the 0-20 year period there is a tighter cluster of more
# active engagement reflecting younger people are almost more likely to click through an ad. 
# In the 65+ period the variance of some high and low CTR seem to suggest that seniors are inconsistent in  
# using the internet to shop. 




