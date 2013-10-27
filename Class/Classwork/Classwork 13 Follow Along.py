# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 19:14:50 2013

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

from sklearn.externals.six.moves import xrange #xrange works like range except it creates an index
from sklearn.tree import DecisionTreeClassifier



