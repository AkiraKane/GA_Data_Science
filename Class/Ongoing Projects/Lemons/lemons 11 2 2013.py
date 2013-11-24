# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 13:09:29 2013

@author: alexandersedgwick
"""

# Load modules
import pandas as pd
from sklearn import tree, datasets, metrics, tree, cross_validation
from matplotlib import pyplot as plt
import datetime
import random
import pandas.tools.rplot as rplot


l_train = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/lemons/lemon_training_mod.csv')
l_test = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/lemons/lemon_test.csv')
l_train = l_train.drop(['PRIMEUNIT'],axis=1)

l_train = l_train.dropna()

#Format Data

l_train = l_train.join(pd.get_dummies(l_train['Auction'], prefix='Auction_'))
l_train = l_train.join(pd.get_dummies(l_train['Color'], prefix='Color_'))
l_train = l_train.join(pd.get_dummies(l_train['Transmission'], prefix='Trans_'))
l_train = l_train.join(pd.get_dummies(l_train['Nationality'], prefix='Natl_'))
l_train = l_train.join(pd.get_dummies(l_train['Size'], prefix='Size_'))
l_train = l_train.join(pd.get_dummies(l_train['TopThreeAmericanName'], prefix='Top3_'))
l_train = l_train.join(pd.get_dummies(l_train['VNST'], prefix='State_'))
l_train = l_train.join(pd.get_dummies(l_train['AUCGUART'], prefix='Guar_'))
l_train = l_train.join(pd.get_dummies(l_train['WheelType'], prefix='Wheels_'))
l_train = l_train.drop(['AUCGUART','WheelType','Auction','Color','Transmission','Nationality','Size','TopThreeAmericanName','VNST'],axis=1)


#Turn purchase date into month and day
l_train['PurchDate'] = pd.to_datetime(l_train['PurchDate'])
l_train['months'] = l_train.PurchDate.map(lambda x: x.month)
l_train['weekdays'] = l_train.PurchDate.map(lambda x: x.weekday())

l_train = l_train.join(pd.get_dummies(l_train['months'], prefix='Month_'))
l_train = l_train.join(pd.get_dummies(l_train['weekdays'], prefix='DayOfWeek_'))
l_train = l_train.drop(['months','weekdays','PurchDate'],axis=1)


# Subset the data so we have a more even data set

lemons = l_train[l_train['IsBadBuy']==1]
non_lemons = l_train[l_train['IsBadBuy']==0]
non_lemons = non_lemons.ix[random.sample(non_lemons.index, 6684)]
train = lemons.append(non_lemons)

#X = train.drop(['RefId','IsBadBuy','VehYear','Make','Model','Trim','SubModel','WheelTypeID','BYRNO'],axis=1)
#y = pd.Series(train['IsBadBuy']).values

target = pd.Series(train['IsBadBuy']).values
data = train.drop(['RefId','IsBadBuy','VehYear','Make','Model','Trim','SubModel','WheelTypeID','BYRNO'],axis=1)

x_train, x_test, y_train, y_test = cross_validation.train_test_split(data,target, test_size=.3)



# Subset the data so we have a more even data set

model = RidgeClassifier()
clf = model.fit(X,y)
Ridg_Class = clf.predict(X)
clf.score(X,y)

metrics.confusion_matrix(y, clf.predict(X))
print metrics.classification_report(y, clf.predict(X))


# GradientBoostingClassifier

from sklearn.ensemble import *
model = GradientBoostingClassifier()

# Train
clf = model.fit(x_train, y_train)

# Get accuracy scores
scores = clf.score(data, target)
metrics.confusion_matrix(y_train, clf.predict(x_train))
print metrics.classification_report(y_train, clf.predict(x_train))
print metrics.auc_score(y_train, clf.predict(x_train))


metrics.confusion_matrix(y_test, clf.predict(x_test))
print metrics.classification_report(y_test, clf.predict(x_test))
print metrics.auc_score(y_test, clf.predict(x_test))

metrics.roc_auc_score(y_train, clf.predict(x_train))
metrics.roc_auc_score(y_test, clf.predict(x_test))



metrics.confusion_matrix(target, clf.predict(data))
print metrics.classification_report(target, clf.predict(data))
print metrics.auc_score(target, clf.predict(data))





##########################################################################################

# Sandbox

##########################################################################################
from sklearn.linear_model import SGDClassifier

sklearn.neighbors.NearestNeighbors
class sklearn.naive_bayes.MultinomialNB:

from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.forest import (RandomForestClassifier,ExtraTreesClassifier)

from sklearn import   
naive_bayes.MultinomialNB
svm
tree.DecisionTreeClassifier
    neighbors.NearestNeighbors
    ensemble.RandomForestClassifier
    ensemble.RandomTreesEmbedding
    ensemble.ExtraTreesClassifier
    ensemble.AdaBoostClassifier
    ensemble.GradientBoostingClassifier
    linear_model.SGDClassifier
    linear_model.RidgeClassifier
    linear_model.RandomizedLogisticRegression
    linear_model.SGDClassifier
    linear_model.LogisticRegression
    
from sklearn.linear_model import RidgeClassifier


#Begin with Logistic Regression

from sklearn.linear_model import (RidgeClassifier,SGDClassifier,LogisticRegression)
from sklearn import svm

#-RidgeClassifier

model = RidgeClassifier()
clf = model.fit(X,y)
Ridg_Class = clf.predict(X)
clf.score(X,y)

metrics.confusion_matrix(y, clf.predict(X))
print metrics.classification_report(y, clf.predict(X))

#-LogisticRegression

model = LogisticRegression()
clf = model.fit(X,y)
Ridg_Class = clf.predict(X)
clf.score(X,y)

metrics.confusion_matrix(y, clf.predict(X))
print metrics.classification_report(y, clf.predict(X))



#-SGD Classifier

model = SGDClassifier()
clf = model.fit(X,y)
SGD_Class = clf.predict(X)
clf.score(X,y)

metrics.confusion_matrix(y, clf.predict(X))
print metrics.classification_report(y, clf.predict(X))


#SVM
model = svm.SVC()
clf = model.fit(X,y)
SVM_Class = clf.predict(X)
clf.score(X,y)









