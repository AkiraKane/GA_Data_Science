# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 21:00:50 2013

@author: alexandersedgwick
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:37:11 2013

@author: alexandersedgwick
"""

# Load modules

import pandas as pd
from sklearn import tree, datasets, metrics, tree, cross_validation

from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.forest import (RandomForestClassifier,
                                        ExtraTreesClassifier)
from sklearn.externals.six.moves import xrange
from matplotlib import pyplot as plt

# Load in data and create training and test sets. dropping all na columns, for kicks.

l_train = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/lemons/lemon_training.csv')
l_test = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/lemons/lemon_test.csv')
l_train = l_train.drop(['PRIMEUNIT','AUCGUART'],axis=1)
l_train = l_train.dropna()
l_test = l_test.dropna()

l_train = l_train[['IsBadBuy','VehYear','Auction','Transmission','WheelType','VehOdo','Size','WarrantyCost']]


#create dummy variables

l_train = l_train.join(pd.get_dummies(l_train['Auction']))
l_train = l_train.join(pd.get_dummies(l_train['Transmission']))
l_train = l_train.join(pd.get_dummies(l_train['WheelType']))
l_train = l_train.join(pd.get_dummies(l_train['Size']))


l_train = l_train.drop(['Auction','Transmission','WheelType','Size'],axis=1)
l_train = l_train.dropna()

data = l_train.drop('IsBadBuy',axis=1)
target = l_train['IsBadBuy']
x_train, x_test, y_train, y_test = cross_validation.train_test_split(data, target, test_size=.3)


# AdaBoost Runs the best

model = AdaBoostClassifier()
clf = model.fit(x_train, y_train)
scores = clf.score(x_train,y_train)

print metrics.classification_report(y_train, clf.predict(x_train))
print metrics.classification_report(y_test, clf.predict(x_test))
y_pred = clf.predict(x_test)

metrics.roc_auc_score(y_train,clf.predict(x_train))
metrics.roc_auc_score(y_test,clf.predict(x_test))

# Create a submission
#submission = pd.DataFrame({ 'RefId' : l_test.RefId, 'prediction' : y_pred })
#submission.to_csv('/users/alexandersedgwick/desktop/submission.csv')

