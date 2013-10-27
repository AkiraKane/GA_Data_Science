# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 18:27:51 2013

@author: alexandersedgwick
"""

import numpy as np
import pylab as pl

from sklearn import clone
from sklearn.datasets import load_iris

# note: these imports are incorrect in the example online!
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.forest import (RandomForestClassifier,
                                        ExtraTreesClassifier)

from sklearn.externals.six.moves import xrange
from sklearn.tree import DecisionTreeClassifier


# Parameters
n_classes = 3
n_estimators = 30
plot_colors = "bry"
plot_step = 0.02

# Load data
iris = load_iris()

plot_idx = 1


## [sepal width, sepal length],[petal legnth, sepal length],[petal width, petal length]

for pair in ([0, 1], [0, 2], [2, 3]):
    for model in (DecisionTreeClassifier(),
                  RandomForestClassifier(n_estimators=n_estimators),
                  ExtraTreesClassifier(n_estimators=n_estimators),
                  AdaBoostClassifier(DecisionTreeClassifier(),
                                     n_estimators=n_estimators)):
        # We only take the two corresponding features
        X = iris.data[:, pair]
        y = iris.target

        # Shuffle
        idx = np.arange(X.shape[0])
        np.random.seed(13)
        np.random.shuffle(idx)
        X = X[idx]
        y = y[idx]

        # Standardize, mean centering the data
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X = (X - mean) / std

        # Train
        clf = model.fit(X, y)

        # Get accuracy scores
        scores = clf.score(X, y)
        # Create a title for each column and the console by using str() and slicing away useless parts of the string
        model_title = str(type(model)).split(".")[-1][:-2][:-len("Classifier")]
        model_details = model_title
        if hasattr(model, "estimators_"):
            model_details += " with {} estimators".format(len(model.estimators_))
        print model_details + " with features", pair, "has a score of", scores

        # Plot the decision boundary
        pl.subplot(3, 4, plot_idx)

        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                             np.arange(y_min, y_max, plot_step))

        if isinstance(model, DecisionTreeClassifier):
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            cs = pl.contourf(xx, yy, Z, cmap=pl.cm.Paired)
        else:
            for tree in model.estimators_:
                Z = tree.predict(np.c_[xx.ravel(), yy.ravel()])
                Z = Z.reshape(xx.shape)
                cs = pl.contourf(xx, yy, Z, alpha=0.1, cmap=pl.cm.Paired)

        pl.axis("tight")

        # Plot the training points
        for i, c in zip(xrange(n_classes), plot_colors):
            idx = np.where(y == i)
            pl.scatter(X[idx, 0], X[idx, 1], c=c, label=iris.target_names[i],
                       cmap=pl.cm.Paired)

        pl.axis("tight")

        plot_idx += 1

pl.suptitle("Decision surfaces of DecisionTreeClassifier, "
            "RandomForestClassifier,\nExtraTreesClassifier"
            " and AdaBoostClassifier")
pl.show()





import random
from pandas import read_csv
from sklearn.cross_validation import train_test_split
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn import metrics
from sklearn import preprocessing
authorship = read_csv('http://people.stern.nyu.edu/jsimonof/AnalCatData/Data/Comma_separated/authorship.csv')
authors = list(set(authorship.Author.values))
le = preprocessing.LabelEncoder()
le.fit(authors)
authorship['Author_num'] = le.transform(authorship['Author'])

#What are some of the stop words we're looking at?
features = list(authorship.columns)
features
features.remove('Author')
features.remove('Author_num')

# Create a random variable (random forests work best with a random variable)
# and create a test and training set
authorship['random'] = [random.random() for i in range(841)]
x_train, x_test, y_train, y_test = train_test_split(authorship[features], authorship.Author_num.values, test_size=0.4, random_state=123)


# Fit Model
etclf = ExtraTreesClassifier(n_estimators=20, max_depth=10, verbose=1)
etclf.fit(x_train, y_train)

# Print Confusion Matrix
metrics.confusion_matrix(etclf.predict(x_test), y_test)


from sklearn.ensemble.forest import RandomForestClassifier

rdclf = RandomForestClassifier(n_estimators=20, max_depth=10)
rdclf.fit(x_train, y_train)
metrics.confusion_matrix(rdclf.predict(x_test), y_test)


from sklearn.ensemble.weight_boosting import AdaBoostClassifier

adaclf = AdaBoostClassifier(n_estimators=20)
adaclf.fit(x_train, y_train)
metrics.confusion_matrix(adaclf.predict(x_test), y_test)




metrics.confusion_matrix(etclf.predict(x_test), y_test)
metrics.confusion_matrix(rdclf.predict(x_test), y_test)
metrics.confusion_matrix(adaclf.predict(x_test), y_test)


#The base random forest model seems to do best here.


import time


start = time.time()
etclf.fit(x_train, y_train)
print 'It took', time.time()-start, 'seconds.'
#.162 seconds


start = time.time()
rdclf.fit(x_train, y_train)
print 'It took', time.time()-start, 'seconds.'
#.135 seconds


start = time.time()
adaclf.fit(x_train, y_train)
print 'It took', time.time()-start, 'seconds.'
#.161 seconds




