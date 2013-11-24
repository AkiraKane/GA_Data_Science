# -*- coding: utf-8 -*-
"""
Created on Fri Nov 08 11:10:12 2013

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
from datetime import datetime, timedelta
    
trades = pd.read_csv('C:/Users/asedgwick/Desktop/ML Bond Pricing/test_round.csv')    

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

#trades = trades.drop(['SHORTNAME','MATURITY','CURAVGPRICE','CURTOTALTRADES','CURTOTALVOLUME'],axis=1)    

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
trades.columns[trades.dtypes == 'object']

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


data.to_csv('C:/Users/asedgwick/Desktop/ML Bond Pricing/clean_round_data.csv')


####Ridge Regression 
from sklearn import linear_model, metrics, tree, cross_validation, ensemble
import matplotlib.pyplot as plt
import statsmodels.api as sm


X= data.drop(['CUSIP','BENCH_YIELD','BID_IDC','CURAVGSPREAD','IMPL_BID_IDC_SPR'],axis=1)
y = data['CURAVGSPREAD'] 

x_train, x_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=.3)

rf = ensemble.RandomForestRegressor()
rf.fit(x_train,y_train)

print 'Train R-sq:',metrics.r2_score(rf.predict(x_train),y_train)
print 'Train MSE:',metrics.mean_squared_error(rf.predict(x_train),y_train)
print 'Test R-sq:',metrics.r2_score(rf.predict(x_test),y_test)
print 'IDC MSE:',metrics.mean_squared_error(data['IMPL_BID_IDC_SPR'], data['CURAVGSPREAD'])
print 'Test MSE:',metrics.mean_squared_error(rf.predict(x_test),y_test)

plt.scatter(data.CURAVGSPREAD,lm.predict(X), cmap=plt.cm.jet,c='blue')
xlabel('TRACE') 
ylabel('Linear Prediction') 
plt.show()

plt.scatter(data.CURAVGSPREAD,rf.predict(X), cmap=plt.cm.jet,c='blue')
xlabel('TRACE') 
ylabel('Random Forest Prediction') 
plt.show()

plt.scatter(data.CURAVGSPREAD,data.IMPL_BID_IDC_SPR, cmap=plt.cm.jet,c='blue')
xlabel('TRACE') 
ylabel('IDC') 
plt.show()



lm = linear_model.Ridge()
lm.fit(x_train,y_train)
print 'R-sq:',metrics.r2_score(lm.predict(X),y)
print 'MSE:',metrics.mean_squared_error(lm.predict(X),y)

#pd.tools.plotting.scatter_matrix(data[['CURAVGSPREAD','IMPL_BID_IDC_SPR','MED_BID_INV','MED_OFF_INV','Close_LQD']], alpha=0.2, diagonal='hist')
#plt.show



import cPickle
# save the classifier
with open('my_dumped_classifier.pkl', 'wb') as fid:
    cPickle.dump(rf, fid)    

# load it again
with open('my_dumped_classifier.pkl', 'rb') as fid:
    gnb_loaded = cPickle.load(fid)



#Nearest Neighbors model regression

###############################################################################
# Fit regression model
from sklearn import neighbors
n_neighbors = 10
knn = neighbors.KNeighborsRegressor(n_neighbors)
model= knn.fit(x_train, y_train)


plt.scatter(data.CURAVGSPREAD,rf.predict(X), cmap=plt.cm.jet,c='blue')
xlabel('TRACE') 
ylabel('Random Forest Prediction') 
plt.show()


#print 'Train R-sq:',metrics.r2_score(model.predict(x_train),y_train)
print 'Train MSE:',metrics.mean_squared_error(model.predict(x_train),y_train)
#print 'Test R-sq:',metrics.r2_score(model.predict(x_test),y_test)
print 'Test MSE:',metrics.mean_squared_error(model.predict(x_test),y_test)
print 'IDC MSE:',metrics.mean_squared_error(data['IMPL_BID_IDC_SPR'], data['CURAVGSPREAD'])




for i, weights in enumerate(['uniform', 'distance']):
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    y_ = knn.fit(X, y).predict(X)

    plt.subplot(2, 1, i + 1)
    plt.scatter(X, y, c='k', label='data')
    plt.plot(X, y_, c='g', label='prediction')
    plt.axis('tight')
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,
                                                               weights))

plt.show()


