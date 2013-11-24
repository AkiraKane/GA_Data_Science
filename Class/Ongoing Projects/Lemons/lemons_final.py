# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 09:16:19 2013

@author: asedgwick
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 13:47:13 2013
@author: asedgwick
"""
# Load modules
import pandas as pd
from sklearn import tree, datasets, metrics, tree, cross_validation
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.forest import (RandomForestClassifier,
ExtraTreesClassifier)
from sklearn.externals.six.moves import xrange
from matplotlib import pyplot as plt
import datetime


#l_train = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/lemons/lemon_training.csv')
#l_test = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/lemons/lemon_test.csv')

l_train = pd.read_csv('C:/lemon_training.csv')
l_train = l_train.drop(['WheelType','VehYear','Trim','RefId','PRIMEUNIT','AUCGUART','SubModel'],axis=1)
l_train = l_train.dropna()



l_train = l_train.join(pd.get_dummies(l_train['Auction'], prefix='Auction_'))
l_train = l_train.join(pd.get_dummies(l_train['Color'], prefix='Color_'))
l_train = l_train.join(pd.get_dummies(l_train['Transmission'], prefix='Trans_'))
l_train = l_train.join(pd.get_dummies(l_train['Nationality'], prefix='Natl_'))
l_train = l_train.join(pd.get_dummies(l_train['Size'], prefix='Size_'))
l_train = l_train.join(pd.get_dummies(l_train['TopThreeAmericanName'], prefix='Top3_'))
l_train = l_train.join(pd.get_dummies(l_train['VNST'], prefix='State_'))
l_train = l_train.drop(['Auction','Color','Transmission','Nationality','Size','TopThreeAmericanName','VNST'],axis=1)


#Look at percentage of lemons by make and model - try to avoid adding ALL combinations as features
data = l_train[['IsBadBuy','Make','Model']]
data['uniqueMake'] = data['Make'] + data['Model']
data = data.drop(['Make','Model'],axis=1)
data = data.groupby('uniqueMake')['IsBadBuy'].agg({'sum': np.sum, 'count' : lambda x: len(x)})
data['percent_lemon'] = data['sum']/data['count']
data.sort('percent_lemon',ascending=False)


#Looking through the data, the following two criteria get me about 100 sig features
keep_features = data[data['count']>30]
keep_features = keep_features[keep_features['percent_lemon']>=.15]
keep_features = keep_features.reset_index()
drop_features = data[data['count']<=30]
drop_features = drop_features.reset_index()


#Add only the Make/Models in data
l_train['uniqueMake'] = l_train['Make'] + l_train['Model']
l_train = l_train.drop(['Make','Model'],axis=1)
l_train = l_train.join(pd.get_dummies(l_train['uniqueMake']))
l_train = l_train.drop(['uniqueMake'],axis=1)
l_train = l_train.drop(drop_features['uniqueMake'],axis=1)

#Turn purchase date into month and day
l_train['PurchDate'] = pd.to_datetime(l_train['PurchDate'])
l_train['months'] = l_train.PurchDate.map(lambda x: x.month)
l_train['weekdays'] = l_train.PurchDate.map(lambda x: x.weekday())

l_train = l_train.join(pd.get_dummies(l_train['months'], prefix='Month_'))
l_train = l_train.join(pd.get_dummies(l_train['weekdays'], prefix='DayOfWeek_'))
l_train = l_train.drop(['months','weekdays','PurchDate'],axis=1)

###Set up test and train data

target = l_train['IsBadBuy']
data = l_train.drop(['IsBadBuy'],axis=1)
x_train, x_test, y_train, y_test = cross_validation.train_test_split(data,target, test_size=.3)

###Random Forest

#from sklearn.ensemble.forest import (RandomForestClassifier,ExtraTreesClassifier)
#from sklearn.ensemble import *
#model = RandomForestClassifier()
#
## Train
#clf = model.fit(x_train, y_train)
#
## Get accuracy scores
#scores = clf.score(data, target)
#metrics.confusion_matrix(y_train, clf.predict(x_train))
#print metrics.classification_report(y_train, clf.predict(x_train))
#
#metrics.confusion_matrix(y_test, clf.predict(x_test))
#print metrics.classification_report(y_test, clf.predict(x_test))
#

from sklearn.ensemble.AdaBoostClassifier import * 
model = GradientBoostingClassifier()

# Train
clf = model.fit(x_train, y_train)

# Get accuracy scores
scores = clf.score(data, target)
metrics.confusion_matrix(y_train, clf.predict(x_train))
print metrics.classification_report(y_train, clf.predict(x_train))

metrics.confusion_matrix(y_test, clf.predict(x_test))
print metrics.classification_report(y_test, clf.predict(x_test))



#### Produce Predicted values

l_test = pd.read_csv('C:/lemon_test.csv')

l_test = l_test.drop(['WheelType','VehYear','Trim','RefId','PRIMEUNIT','AUCGUART','SubModel'],axis=1)
l_test = l_test.dropna()

l_test = l_test.join(pd.get_dummies(l_test['Auction'], prefix='Auction_'))
l_test = l_test.join(pd.get_dummies(l_test['Color'], prefix='Color_'))
l_test = l_test.join(pd.get_dummies(l_test['Transmission'], prefix='Trans_'))
l_test = l_test.join(pd.get_dummies(l_test['Nationality'], prefix='Natl_'))
l_test = l_test.join(pd.get_dummies(l_test['Size'], prefix='Size_'))
l_test = l_test.join(pd.get_dummies(l_test['TopThreeAmericanName'], prefix='Top3_'))
l_test = l_test.join(pd.get_dummies(l_test['VNST'], prefix='State_'))
l_test = l_test.drop(['Auction','Color','Transmission','Nationality','Size','TopThreeAmericanName','VNST'],axis=1)

l_test['uniqueMake'] = l_test['Make'] + l_test['Model']
l_test = l_test.drop(['Make','Model'],axis=1)
l_test = l_test.join(pd.get_dummies(l_test['uniqueMake']))
l_test = l_test.drop(['uniqueMake'],axis=1)
l_test = l_test.drop(drop_features['uniqueMake'],axis=1)

l_test['PurchDate'] = pd.to_datetime(l_test['PurchDate'])
l_test['months'] = l_test.PurchDate.map(lambda x: x.month)
l_test['weekdays'] = l_test.PurchDate.map(lambda x: x.weekday())

l_test = l_test.join(pd.get_dummies(l_test['months'], prefix='Month_'))
l_test = l_test.join(pd.get_dummies(l_test['weekdays'], prefix='DayOfWeek_'))
l_test = l_test.drop(['months','weekdays','PurchDate'],axis=1)

clf.predict(x_train)