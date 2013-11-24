# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 08:02:20 2013

@author: asedgwick
"""


import pandas as pd
import numpy as np
import pandas.io.sql as psql
from pandas.io.data import DataReader
import pandasql as pysqldf
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import statsmodels.api as sm
from sklearn.preprocessing import scale
from sklearn import metrics

    
trades = pd.read_csv('C:/Users/asedgwick/Desktop/ML Bond Pricing/test_data.csv')    
trades = trades.dropna()    
    
#colsolidate a few of the sectors
trades['TRADING_SECTOR'][trades['TRADING_SECTOR'].isin(['Asia IG','Unassigned','Latam Corps'])]='Other'
trades['INDUSTRY'] = 'Industrial'
trades['INDUSTRY'][trades['TRADING_SECTOR'].isin(['Banks/Finance','Insurance',''])]='Finance'
trades['INDUSTRY'][trades['TRADING_SECTOR'].isin(['Utility'])]='Utility'
#Drop Supranationals
trades = trades[trades['TRADING_SECTOR']!='Supranational']


#Bucket the Ratings Dimport pandas as pd
trades['RatingBucket'] = 'Other'	
trades['RatingBucket'][trades['SNPRATING']==1] = 'AAA'
trades['RatingBucket'][(trades['SNPRATING']>=2) & (trades['SNPRATING']<=4)] = 'AA'
trades['RatingBucket'][(trades['SNPRATING']>=5) & (trades['SNPRATING']<=7)] = 'A'
trades['RatingBucket'][(trades['SNPRATING']>=8) & (trades['SNPRATING']<=10)] = 'BBB'
trades['RatingBucket'][(trades['SNPRATING']>=11) & (trades['SNPRATING']<=13)] = 'BB'
trades['RatingBucket'][(trades['SNPRATING']>=14) & (trades['SNPRATING']<=16)] = 'B'
trades['RatingBucket'][(trades['SNPRATING']>=17) & (trades['SNPRATING']<=19)] = 'CCC'
trades['RatingBucket'][trades['SNPRATING']==20] = 'CC'
trades['RatingBucket'][trades['SNPRATING']==21] = 'C'
trades['RatingBucket'][trades['SNPRATING']==25] = 'D'
trades = trades.join(pd.get_dummies(trades['RatingBucket'], prefix='Rat'))
trades = trades.drop(['RatingBucket','SNPRATING','SNPRATINGDISP'], axis=1)

#Bucket Issue Amount
trades['IssueBucket'] = '<250 MM'
trades['IssueBucket'][(trades['ISSUEAMOUNT']>=250000000) & (trades['ISSUEAMOUNT']<500000000)] = '250-500MM'
trades['IssueBucket'][(trades['ISSUEAMOUNT']>=500000000) & (trades['ISSUEAMOUNT']<1000000000)] = '500-1BN'
trades['IssueBucket'][trades['ISSUEAMOUNT']>=1000000000] = '1BN+'
del trades['ISSUEAMOUNT']

trades = trades.join(pd.get_dummies(trades['IssueBucket'], prefix='ISS'))
trades = trades.drop(['IssueBucket'], axis=1)

#Determine Bond Age
trades['CURRDATE'] = pd.to_datetime(trades['CURRDATE'])    
trades['ISSUEDATE'] = pd.to_datetime(trades['ISSUEDATE'])    
trades['Age'] = (trades['CURRDATE'] - trades['ISSUEDATE']).values/np.timedelta64(1,'D')
del trades['ISSUEDATE']

trades = trades.drop(['SHORTNAME','MATURITY','CURAVGPRICE','CURTOTALTRADES','CURTOTALVOLUME'],axis=1)    

#Create Dummy Variables for Maturity and sector
trades = trades.join(pd.get_dummies(trades['MATURITY_BUCKET'], prefix='MAT'))
trades = trades.drop(['MATURITY_BUCKET'], axis=1)
trades = trades.join(pd.get_dummies(trades['TRADING_SECTOR'], prefix='SEC'))
trades = trades.drop(['TRADING_SECTOR'], axis=1)
trades = trades.join(pd.get_dummies(trades['INDUSTRY'], prefix='Ind'))
trades = trades.drop(['INDUSTRY'], axis=1)
trades = trades.join(pd.get_dummies(trades['DEFAULTTICKER']))
trades = trades.drop(['DEFAULTTICKER'], axis=1)

#Code Bonds as fixed or floating rate
trades['FIXED']=0
trades['FIXED'][trades['INTERESTRATETYPE']=='F'] = 1
del trades['INTERESTRATETYPE']

trades['MARKET'] = 'OTHER'
trades['MARKET'][trades['MARKETSEGMENT']==4] = 'HG'
trades['MARKET'][trades['MARKETSEGMENT']==7] = 'FRN'
trades['MARKET'][trades['MARKETSEGMENT']==0] = 'HY'
trades['MARKET'][trades['MARKETSEGMENT']==8] = 'AGY'

trades = trades.join(pd.get_dummies(trades['MARKET'], prefix='MKT'))
trades = trades.drop(['MARKET'], axis=1)
del trades['ISCVTPREFSTOCK']
del trades['MARKETSEGMENT']
#trades.columns[trades.dtypes == 'object']

#Pull ETF data in liey of indices
#Scrape from Yahoo
LQD = DataReader("LQD",  "yahoo", datetime(2013,1,1), datetime(2013,12,31))
LQD['Date'] = LQD.index
LQD = LQD.reset_index(drop=True)
#Get the vix
VIX = DataReader("VIX",  "yahoo", datetime(2013,1,1), datetime(2013,12,31))
VIX['Date'] = VIX.index
VIX = VIX.reset_index(drop=True)
stocks =pd.merge(VIX, LQD, left_on='Date', right_on='Date', how='outer', suffixes=('_VIX','_LQD') )
stocks=stocks[['Date','Close_VIX','Close_LQD']]

#Get Treasury Rates
UST_2 = DataReader("DGS2",  "fred", datetime(2013,1,1), datetime(2013,12,31))
UST_10 = DataReader("DGS10",  "fred", datetime(2013,1,1), datetime(2013,12,31))
UST_30 = DataReader("DGS30",  "fred", datetime(2013,1,1), datetime(2013,12,31))
UST = pd.DataFrame({'Date': UST_2['DGS2'].index.values,'UST2': UST_2['DGS2'].values ,'UST10': UST_10['DGS10'].values,'UST30': UST_30['DGS30'].values})

ex_data =pd.merge(stocks, UST, left_on='Date', right_on='Date', how='outer', suffixes=('_ST','_UST') )
ex_data = ex_data.dropna()
data =pd.merge(trades, ex_data, left_on='CURRDATE', right_on='Date', how='outer', suffixes=('_BND','_STK') )

data = data.drop(['Date','CURRDATE'],axis=1)

del ex_data
del UST_2
del UST_10
del UST_30
del UST
del VIX
del LQD
del stocks

data = data[data.UST30!='.']
data['UST2'] = data['UST2'].astype(np.float)
data['UST10'] = data['UST10'].astype(np.float)
data['UST30'] = data['UST30'].astype(np.float)


# Filter the data
data = data[data['CURAVGSPREAD']>=0]
data = data[data['IMPL_BID_IDC_SPR']>=0]
data = data[data['CURAVGSPREAD']<=750]
data = data[data['IMPL_BID_IDC_SPR']<=750]
data = data[(data['MED_BID_INV']-data['MED_OFF_INV']<100) & (data['MED_BID_INV']-data['MED_OFF_INV']>-100)]

data = data.replace([inf, -inf], np.nan)
data = data.dropna()


data.to_csv('C:/Users/asedgwick/Desktop/ML Bond Pricing/clean_data_11_12_2013.csv')

data = pd.read_csv('C:/Users/asedgwick/Desktop/ML Bond Pricing/clean_data_11_12_2013.csv')

from ggplot import *

ggplot(aes(x='CURAVGSPREAD', y='IMPL_BID_IDC_SPR', alpha=0.5), data=data) + \
    geom_point() + \
    ggtitle("IDC vs. TRACE Spreads") + \
    xlab("TRACE Observed Spread (bps)") + \
    ylab("Evaluated Spread (bps)")


#plt.scatter(data.CURAVGSPREAD,data.IMPL_BID_IDC_SPR, cmap=plt.cm.jet,c='blue')
#xlabel('Observed TRACE Spread (bps)') 
#ylabel('Evaluated IDC Spread (bps)') 
#plt.show()



####Random Forest Model
from sklearn import linear_model, metrics, tree, cross_validation, ensemble
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor


#Random Forest without IDC

X= data.drop(['CUSIP','BENCH_YIELD','BID_IDC','CURAVGSPREAD','IMPL_BID_IDC_SPR'],axis=1)
y = data['CURAVGSPREAD'] 

x_train, x_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=.3)

rf1 = ensemble.RandomForestRegressor()
rf1.fit(x_train,y_train)

predicted = rf1.predict(x_train)

print 'Train R-sq:',metrics.r2_score(rf1.predict(x_train),y_train)
print 'Train MSE:',metrics.mean_squared_error(rf1.predict(x_train),y_train)
print 'Test R-sq:',metrics.r2_score(rf1.predict(x_test),y_test)
print 'IDC MSE:',metrics.mean_squared_error(data['IMPL_BID_IDC_SPR'], data['CURAVGSPREAD'])
print 'Test MSE:',metrics.mean_squared_error(rf1.predict(x_test),y_test)

graph_data = pd.DataFrame({'Observed':y,'RandForest':rf1.predict(X)})


ggplot(aes(x='Observed', y='RandForest', alpha=0.5), data=graph_data) + \
    geom_point() + \
    ggtitle("Random Forest Model ex IDC vs. TRACE Spreads") + \
    xlab("TRACE Observed Spread (bps)") + \
    ylab("Random Forest Prediction")
    
    
    
graph_data = pd.DataFrame({'Observed':data.CURAVGSPREAD,'IDC':data.IMPL_BID_IDC_SPR})


ggplot(aes(x='Observed', y='IDC', alpha=0.5), data=graph_data) + \
    geom_point() + \
    ggtitle("Evaluated Price vs. TRACE Spreads") + \
    xlab("TRACE Observed Spread (bps)") + \
    ylab("Evaluated Sprad (bps)")

predictions = pd.DataFrame({'Observed':data['CURAVGSPREAD'],'RandomForest':rf1.predict(X),'IDC':data['IMPL_BID_IDC_SPR'] })


melt = pd.melt(predictions[['IDCErr','RFErr']])

p = ggplot(aes(x='value', colour='variable', fill=True, alpha=0.3), data=melt)
p + geom_density() + \
    xlim(-100,100) + \
    ggtitle("Pricing Error Frequency: Evaluated vs. Random Forest Model")

##Random Forest with IDC
#
#X= data.drop(['CUSIP','BENCH_YIELD','BID_IDC','CURAVGSPREAD'],axis=1)
#y = data['CURAVGSPREAD'] 
#
#x_train, x_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=.3)
#
#rf2 = ensemble.RandomForestRegressor()
#rf2.fit(x_train,y_train)
#
#predicted = rf2.predict(x_train)
#
#print 'Train R-sq:',metrics.r2_score(rf2.predict(x_train),y_train)
#print 'Train MSE:',metrics.mean_squared_error(rf2.predict(x_train),y_train)
#print 'Test R-sq:',metrics.r2_score(rf2.predict(x_test),y_test)
#print 'IDC MSE:',metrics.mean_squared_error(data['IMPL_BID_IDC_SPR'], data['CURAVGSPREAD'])
#print 'Test MSE:',metrics.mean_squared_error(rf2.predict(x_test),y_test)
#
#
#graph_data = pd.DataFrame({'Observed':y,'RandForest':rf2.predict(X)})
#
#
#ggplot(aes(x='Observed', y='RandForest', alpha=0.5), data=graph_data) + \
#    geom_point() + \
#    ggtitle("Random Forest Model ex IDC vs. TRACE Spreads") + \
#    xlab("TRACE Observed Spread (bps)") + \
#    ylab("Random Forest Prediction")


#

tree1 = tree.DecisionTreeRegressor()
tree1.fit(x_train,y_train)
print 'Test MSE:',metrics.mean_squared_error(tree1.predict(x_test),y_test)
#MSE: 190

et1 = tree.ExtraTreeRegressor()
et1.fit(x_train,y_train)

print 'Test MSE:',metrics.mean_squared_error(et1.predict(x_test),y_test)
#MSE: 178

gb1 = ensemble.GradientBoostingRegressor()
gb1.fit(x_train,y_train)

print 'Test MSE:',metrics.mean_squared_error(gb1.predict(x_test),y_test)
#MSE = 149


from sklearn import svm
svm = svm.SVR()
svm.fit(x_train,y_train)
print 'Test MSE:',metrics.mean_squared_error(svm.predict(x_test),y_test)
#MSE: 190

lm = linear_model.Ridge (alpha = .5)
lm.fit(X,y)
print 'Test MSE:',metrics.mean_squared_error(lm.predict(x_test),y_test)


svm = svm.SVR()
svm.fit(x_train,y_train)
print 'Test MSE:',metrics.mean_squared_error(svm.predict(x_test),y_test)
#MSE: 190



kaggle = pd.read_csv('C:/Users/asedgwick/Desktop/ML Bond Pricing/training_data_vix.csv')
kaggle = kaggle.drop(['CUSIP','price','ticker','coupon','maturity','issue_date','callable','iscvtprefstock','putable','trading_sector','snprating','isfdicbond','date','date_1','date_2','date_3','date_4','date_5','date_6','date_7','date_8','date_9','date_10'], axis=1)
kaggle.drop([''], axis=1)

kaggle['reportingpartyside'][kaggle['reportingpartyside']=='B'] = 0
kaggle['reportingpartyside'][kaggle['reportingpartyside']=='S'] = 1
kaggle['reportingpartyside'][kaggle['reportingpartyside']=='D'] = 2

kaggle['reportingpartyside_last_1'][kaggle['reportingpartyside_last_1']=='B'] = 0
kaggle['reportingpartyside_last_1'][kaggle['reportingpartyside_last_1']=='S'] = 1
kaggle['reportingpartyside_last_1'][kaggle['reportingpartyside_last_1']=='D'] = 2

kaggle['reportingpartyside_last_2'][kaggle['reportingpartyside_last_2']=='B'] = 0
kaggle['reportingpartyside_last_2'][kaggle['reportingpartyside_last_2']=='S'] = 1
kaggle['reportingpartyside_last_2'][kaggle['reportingpartyside_last_2']=='D'] = 2

kaggle['reportingpartyside_last_3'][kaggle['reportingpartyside_last_3']=='B'] = 0
kaggle['reportingpartyside_last_3'][kaggle['reportingpartyside_last_3']=='S'] = 1
kaggle['reportingpartyside_last_3'][kaggle['reportingpartyside_last_3']=='D'] = 2

kaggle['reportingpartyside_last_4'][kaggle['reportingpartyside_last_4']=='B'] = 0
kaggle['reportingpartyside_last_4'][kaggle['reportingpartyside_last_4']=='S'] = 1
kaggle['reportingpartyside_last_4'][kaggle['reportingpartyside_last_4']=='D'] = 2

kaggle['reportingpartyside_last_5'][kaggle['reportingpartyside_last_5']=='B'] = 0
kaggle['reportingpartyside_last_5'][kaggle['reportingpartyside_last_5']=='S'] = 1
kaggle['reportingpartyside_last_5'][kaggle['reportingpartyside_last_5']=='D'] = 2

kaggle['reportingpartyside_last_6'][kaggle['reportingpartyside_last_6']=='B'] = 0
kaggle['reportingpartyside_last_6'][kaggle['reportingpartyside_last_6']=='S'] = 1
kaggle['reportingpartyside_last_6'][kaggle['reportingpartyside_last_6']=='D'] = 2

kaggle['reportingpartyside_last_7'][kaggle['reportingpartyside_last_7']=='B'] = 0
kaggle['reportingpartyside_last_7'][kaggle['reportingpartyside_last_7']=='S'] = 1
kaggle['reportingpartyside_last_7'][kaggle['reportingpartyside_last_7']=='D'] = 2

kaggle['reportingpartyside_last_8'][kaggle['reportingpartyside_last_8']=='B'] = 0
kaggle['reportingpartyside_last_8'][kaggle['reportingpartyside_last_8']=='S'] = 1
kaggle['reportingpartyside_last_8'][kaggle['reportingpartyside_last_8']=='D'] = 2

kaggle['reportingpartyside_last_9'][kaggle['reportingpartyside_last_9']=='B'] = 0
kaggle['reportingpartyside_last_9'][kaggle['reportingpartyside_last_9']=='S'] = 1
kaggle['reportingpartyside_last_9'][kaggle['reportingpartyside_last_9']=='D'] = 2

kaggle['reportingpartyside_last_10'][kaggle['reportingpartyside_last_10']=='B'] = 0
kaggle['reportingpartyside_last_10'][kaggle['reportingpartyside_last_10']=='S'] = 1
kaggle['reportingpartyside_last_10'][kaggle['reportingpartyside_last_10']=='D'] = 2


kaggle['quantitythreshhold'][kaggle['quantitythreshhold']=='Y'] = 1
kaggle['quantitythreshhold'][kaggle['quantitythreshhold']=='N'] = 0

kaggle['quantitythreshhold_last_1'][kaggle['quantitythreshhold_last_1']=='Y'] = 1
kaggle['quantitythreshhold_last_1'][kaggle['quantitythreshhold_last_1']=='N'] = 0


kaggle['quantitythreshhold_last_2'][kaggle['quantitythreshhold_last_2']=='Y'] = 1
kaggle['quantitythreshhold_last_2'][kaggle['quantitythreshhold_last_2']=='N'] = 0

kaggle['quantitythreshhold_last_3'][kaggle['quantitythreshhold_last_3']=='Y'] = 1
kaggle['quantitythreshhold_last_3'][kaggle['quantitythreshhold_last_3']=='N'] = 0

kaggle['quantitythreshhold_last_4'][kaggle['quantitythreshhold_last_4']=='Y'] = 1
kaggle['quantitythreshhold_last_4'][kaggle['quantitythreshhold_last_4']=='N'] = 0

kaggle['quantitythreshhold_last_5'][kaggle['quantitythreshhold_last_5']=='Y'] = 1
kaggle['quantitythreshhold_last_5'][kaggle['quantitythreshhold_last_5']=='N'] = 0

kaggle['quantitythreshhold_last_6'][kaggle['quantitythreshhold_last_6']=='Y'] = 1
kaggle['quantitythreshhold_last_6'][kaggle['quantitythreshhold_last_6']=='N'] = 0

kaggle['quantitythreshhold_last_7'][kaggle['quantitythreshhold_last_7']=='Y'] = 1
kaggle['quantitythreshhold_last_7'][kaggle['quantitythreshhold_last_7']=='N'] = 0

kaggle['quantitythreshhold_last_8'][kaggle['quantitythreshhold_last_8']=='Y'] = 1
kaggle['quantitythreshhold_last_8'][kaggle['quantitythreshhold_last_8']=='N'] = 0

kaggle['quantitythreshhold_last_9'][kaggle['quantitythreshhold_last_9']=='Y'] = 1
kaggle['quantitythreshhold_last_9'][kaggle['quantitythreshhold_last_9']=='N'] = 0

kaggle['quantitythreshhold_last_10'][kaggle['quantitythreshhold_last_10']=='Y'] = 1
kaggle['quantitythreshhold_last_10'][kaggle['quantitythreshhold_last_10']=='N'] = 0


kaggle = kaggle.drop(['spread_last_8','spread_last_9','spread_last_10'],axis=1)



X= kaggle.drop(['oasspread'],axis=1)
y = kaggle['oasspread'] 

x_train, x_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=.3)

rf1 = ensemble.RandomForestRegressor()
rf1.fit(x_train,y_train)

predicted = rf1.predict(x_train)

print 'Train R-sq:',metrics.r2_score(rf1.predict(x_train),y_train)
print 'Train MSE:',metrics.mean_squared_error(rf1.predict(x_train),y_train)
print 'Test R-sq:',metrics.r2_score(rf1.predict(x_test),y_test)
print 'Test MSE:',metrics.mean_squared_error(rf1.predict(x_test),y_test)


kaggle = kaggle.dropna()



