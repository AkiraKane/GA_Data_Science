# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:37:11 2013

@author: alexandersedgwick
"""

# Load modules

import pandas as pd
from sklearn import tree, datasets, metrics, tree, cross_validation
from matplotlib import pyplot as plt

# Load in data and create training and test sets. dropping all na columns, for kicks.

l_train = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/lemons/lemon_training.csv')
l_test = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/lemons/lemon_test.csv')
l_train = l_train.drop(['PRIMEUNIT','AUCGUART'],axis=1)
l_train = l_train.dropna()
l_test = l_test.dropna()

l_train = l_train[['IsBadBuy','VehYear','Auction','Transmission','WheelType','VehOdo','Size','WarrantyCost']]



#histogram
#pd.tools.plotting.scatter_matrix(l_train, alpha=0.2, diagonal='hist')
#plt.show()

#create dummy variables

l_train = l_train.join(pd.get_dummies(l_train['Auction']))
l_train = l_train.join(pd.get_dummies(l_train['Transmission']))
l_train = l_train.join(pd.get_dummies(l_train['WheelType']))
l_train = l_train.join(pd.get_dummies(l_train['Size']))


l_train = l_train.drop(['Auction','Transmission','WheelType','Size'],axis=1)
l_train = l_train.dropna()

#l_train['Size'].unique()

#data=l_train.drop(['IsBadBuy'],axis=1)

data = l_train.drop('IsBadBuy',axis=1)
target = l_train['IsBadBuy']
x_train, x_test, y_train, y_test = cross_validation.train_test_split(data, target, test_size=.3)


#Naive Bayes

from sklearn import datasets, metrics
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_output = nb_model.fit(data, target).predict(data)
print("Number of mislabeled points : %d" % (target != nb_output).sum())

# Finding the false positive and true positive rates where the positive label is 2.
fpr, tpr, thresholds = metrics.roc_curve(target, nb_output, pos_label=2)
metrics.auc(fpr, tpr)
plt.plot(fpr, tpr)
plt.show()



#Decision Trees

from sklearn import datasets, metrics, tree, cross_validation
from matplotlib import pyplot as plt

y_pred = tree.DecisionTreeClassifier().fit(data, target).predict(data)
print("Number of mislabeled points : %d" % (target != y_pred).sum())
print("Absolutely ridiculously overfit score: %d" % (tree.DecisionTreeClassifier().fit(data, target).score(data, target)))

clf = tree.DecisionTreeClassifier()

x_train, x_test, y_train, y_test = cross_validation.train_test_split(data, target, test_size=.3)
clf.fit(x_train, y_train)


metrics.confusion_matrix(y_train, clf.predict(x_train))
print metrics.classification_report(y_train, clf.predict(x_train))
metrics.confusion_matrix(y_test, clf.predict(x_test))
print metrics.classification_report(y_test, clf.predict(x_test))


clf.set_params(min_samples_leaf=5)
clf.set_params(max_depth=5)
clf.fit(x_train, y_train)
metrics.confusion_matrix(y_train, clf.predict(x_train))
metrics.confusion_matrix(y_test, clf.predict(x_test))

print metrics.classification_report(y_train, clf.predict(x_train))
print metrics.classification_report(y_test, clf.predict(x_test))

#Cross validation fails, I only find .18 of out of sample lemons



# AdaBoost and Random Forest models

from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.forest import (RandomForestClassifier,
                                        ExtraTreesClassifier)

from sklearn.externals.six.moves import xrange



# AdaBoost Runs the best

model = AdaBoostClassifier()
clf = model.fit(x_train, y_train)
scores = clf.score(x_train,y_train)

print metrics.classification_report(y_train, clf.predict(x_train))
print metrics.classification_report(y_test, clf.predict(x_test))




model = RandomForestClassifier()
clf = model.fit(x_train, y_train)
scores = clf.score(x_train,y_train)

print metrics.classification_report(y_train, clf.predict(x_train))
print metrics.classification_report(y_test, clf.predict(x_test))


model = ExtraTreesClassifier()
clf = model.fit(x_train, y_train)
scores = clf.score(x_train,y_train)

print metrics.classification_report(y_train, clf.predict(x_train))
print metrics.classification_report(y_test, clf.predict(x_test))

