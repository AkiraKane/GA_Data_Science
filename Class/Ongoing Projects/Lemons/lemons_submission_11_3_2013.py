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


##### GradientBoostingClassifier

from sklearn.ensemble import *
model = GradientBoostingClassifier()

# Train
clf = model.fit(x_train, y_train)

# Get accuracy scores
scores = clf.score(x_train, y_train)
metrics.confusion_matrix(y_train, clf.predict(x_train))
print metrics.classification_report(y_train, clf.predict(x_train))
print metrics.auc_score(y_train, clf.predict(x_train))


metrics.confusion_matrix(y_test, clf.predict(x_test))
print metrics.classification_report(y_test, clf.predict(x_test))
print metrics.auc_score(y_test, clf.predict(x_test))

metrics.roc_auc_score(y_train, clf.predict(x_train))
metrics.roc_auc_score(y_test, clf.predict(x_test))

metrics.mean_squared_error(y_train,clf.predict(x_train))
metrics.mean_squared_error(y_test,clf.predict(x_test))



#Prep submission
l_test = pd.read_csv('/users/alexandersedgwick/dropbox/development/ga/Ongoing Projects/lemons/lemon_test.csv')

l_test['AUCGUART'] = l_test['AUCGUART'].fillna('YELLOW')
l_test = l_test.drop(['PRIMEUNIT'],axis=1)
l_test = l_test.join(pd.get_dummies(l_test['Auction'], prefix='Auction_'))
l_test = l_test.join(pd.get_dummies(l_test['Color'], prefix='Color_'))
l_test = l_test.join(pd.get_dummies(l_test['Transmission'], prefix='Trans_'))
l_test = l_test.join(pd.get_dummies(l_test['Nationality'], prefix='Natl_'))
l_test = l_test.join(pd.get_dummies(l_test['Size'], prefix='Size_'))
l_test = l_test.join(pd.get_dummies(l_test['TopThreeAmericanName'], prefix='Top3_'))
l_test = l_test.join(pd.get_dummies(l_test['VNST'], prefix='State_'))
l_test = l_test.join(pd.get_dummies(l_test['AUCGUART'], prefix='Guar_'))
l_test = l_test.join(pd.get_dummies(l_test['WheelType'], prefix='Wheels_'))
l_test = l_test.drop(['AUCGUART','WheelType','Auction','Color','Transmission','Nationality','Size','TopThreeAmericanName','VNST'],axis=1)
l_test['PurchDate'] = pd.to_datetime(l_test['PurchDate'])
l_test['months'] = l_test.PurchDate.map(lambda x: x.month)
l_test['weekdays'] = l_test.PurchDate.map(lambda x: x.weekday())
l_test = l_test.join(pd.get_dummies(l_test['months'], prefix='Month_'))
l_test = l_test.join(pd.get_dummies(l_test['weekdays'], prefix='DayOfWeek_'))
l_test = l_test.drop(['months','weekdays','PurchDate'],axis=1)
l_test = l_test.dropna()

data2 = l_test.drop(['RefId','VehYear','Make','Model','Trim','SubModel','WheelTypeID','BYRNO'],axis=1)
data2['DayOfWeek__5']=0

set(data.columns)-set(data2.columns)

y_pred = clf.predict(data2)

# Create a submission
submission = pd.DataFrame({ 'RefId' : l_test.RefId, 'prediction' : y_pred })
submission.to_csv('/users/alexandersedgwick/dropbox/development/ga/ongoing projects/lemons/lemons_submission 11 4 2013.csv')






