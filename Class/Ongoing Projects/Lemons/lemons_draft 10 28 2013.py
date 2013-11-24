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

l_train = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/lemons/lemon_training.csv')
l_test = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/lemons/lemon_test.csv')
l_train = l_train.drop(['WheelType','VehYear','Trim','RefId','PRIMEUNIT','AUCGUART','SubModel'],axis=1)
l_train = l_train.dropna()
l_test = l_test.dropna()

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
l_train = l_train.drop(drop_features['uniqueMake'],axis=1)
l_train = l_train.drop(['uniqueMake'],axis=1)


#Turn purchase date into month and day
l_train['PurchDate'] = pd.to_datetime(l_train['PurchDate'])
l_train['months'] = l_train.PurchDate.map(lambda x: x.month)
l_train['weekdays'] = l_train.PurchDate.map(lambda x: x.weekday())

l_train = l_train.join(pd.get_dummies(l_train['months'], prefix='Month_'))
l_train = l_train.join(pd.get_dummies(l_train['weekdays'], prefix='DayOfWeek_'))
l_train = l_train.drop(['months','weekdays','PurchDate'],axis=1)





from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0,compute_importances=True)

target = df['IsBadBuy']
data = df.drop(['IsBadBuy'],axis=1)


model = clf.fit(data,target)
y_pred = model.predict(data)

metrics.confusion_matrix(target, y_pred)
print metrics.classification_report(target, y_pred)

test = pd.DataFrame(data.columns)
test2 = pd.DataFrame(model.feature_importances_)
test=test.unstack()
test2 = test2.unstack()

test = pd.concat([test,test2],axis=1)
test =test.sort(1,ascending=False)



# trim features based on decision tree and gini importance

from sklearn.cross_validation import cross_val_score
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=0,compute_importances=True)

target = l_train['IsBadBuy']
data = l_train.drop(['IsBadBuy'],axis=1)


model = clf.fit(data,target)
y_pred = model.predict(data)

metrics.confusion_matrix(target, y_pred)
print metrics.classification_report(target, y_pred)

test = pd.DataFrame(data.columns)
test2 = pd.DataFrame(model.feature_importances_)
test=test.unstack()
test2 = test2.unstack()

test = pd.concat([test,test2],axis=1)
test =test.sort(1,ascending=False)

test.to_csv('C:/feature_selection.csv')

#most important variables based on my decision tree
data = data[['VehOdo','MMRAcquisitionAuctionAveragePrice','VehBCost','MMRCurrentAuctionAveragePrice','MMRAcquisitionRetailAveragePrice','VehicleAge','MMRAcquisitionAuctionCleanPrice','BYRNO','MMRCurrentAuctionCleanPrice','MMRCurrentRetailAveragePrice','MMRAcquisitonRetailCleanPrice','WarrantyCost','WheelTypeID','MMRCurrentRetailCleanPrice']]



#Trim Tree
x_train, x_test, y_train, y_test = cross_validation.train_test_split(data,target, test_size=.3)
model2 = clf.fit(x_train, y_train)

metrics.confusion_matrix(y_train, model2.predict(x_train))
print metrics.classification_report(y_train, model2.predict(x_train))

metrics.confusion_matrix(y_test, model2.predict(x_test))
print metrics.classification_report(y_test, model2.predict(x_test))

clf.set_params(min_samples_leaf=5)
clf.set_params(max_depth=5)
model3 = clf.fit(x_train, y_train)
metrics.confusion_matrix(y_train, model3.predict(x_train))
print metrics.classification_report(y_train, model3.predict(x_train))

metrics.confusion_matrix(y_test, model3.predict(x_test))
print metrics.classification_report(y_train, model3.predict(x_train))


#

from sklearn.ensemble.forest import (RandomForestClassifier,
                                        ExtraTreesClassifier)
from sklearn.ensemble import *

clf = RandomForestClassifier()

 # Train

x_train, x_test, y_train, y_test = cross_validation.train_test_split(data,target, test_size=.3)

clf = model.fit(x_train, y_train)

 # Get accuracy scores
scores = clf.score(data, target)
metrics.confusion_matrix(y_test, clf.predict(x_test))

print metrics.classification_report(y_test, clf.predict(x_test))