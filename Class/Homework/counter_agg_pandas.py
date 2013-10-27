# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 19:18:03 2013

@author: alexandersedgwick
"""
import numpy as numpy
import pandas as pd

df = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/data/nytimes.csv')

df
df.describe()
df[:10]

# Create the average impressions and clicks for each age.
dfg = df[ ['Age', 'Impressions', 'Clicks'] ].groupby(['Age']).agg([numpy.mean])
dfg[:10]


df['log_impressions'] = df['Impressions'].apply(numpy.log)


# Function that groups users by age.
def map_age_category(x):
    if x < 18:
        return '1'
    elif x < 25:
        return '2'
    elif x < 32:
        return '3'
    elif x < 45:
        return '4'
    else:
        return '5'

df['age_categories'] = df['Age'].apply(map_age_category)