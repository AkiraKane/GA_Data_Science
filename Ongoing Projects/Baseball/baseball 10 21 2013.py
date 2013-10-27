# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 20:20:11 2013

@author: alexandersedgwick
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import *
from sklearn.feature_selection import SelectPercentile, f_classif
import statsmodels.api as sm
import locale
locale.setlocale(locale.LC_ALL, 'en_US')


#Load the data
b2011 = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/baseball/baseball_training_2011.csv')
b2012 = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/baseball/baseball_test_2012.csv')



#Identify the different data types
baseball_train.dtypes

train_X = b2011[['G', 'AB', 'R', 'H', 'X2B', 'X3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF']]
train_y = b2011['salary']

#
test_X = b2012[['G', 'AB', 'R', 'H', 'X2B', 'X3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF']].values
b2012_csv = b2012[['playerID','yearID', 'salary']]


#Which teams pay a lot
grouped = b2011.groupby('teamID')
salary_list = grouped['salary'].aggregate(np.sum)
salary_list.sort(1,ascending=False)


