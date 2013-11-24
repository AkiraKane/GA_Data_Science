# -*- coding: utf-8 -*-
"""
Created on Fri Nov 08 11:10:12 2013

@author: asedgwick
"""


import pandas as pd
import numpy as np
import pandas.io.sql as psql
import pandasql as pysqldf
import pandas.io.sql as psql
import cx_Oracle
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn import metrics
from datetime import datetime, timedelta

#Set up a statement to return sql queries
def pysql(q):
    return sqldf(q, globals())


con = cx_Oracle.connect('etf_user/etf_user@PRDPREP.world')


try:
    con = cx_Oracle.connect('etf_user/etf_user@PRDPREP.world')
except cx_Oracle.DatabaseError, e:
    print e[0].context
    raise

sql = """
    select /*+ PARALLEL */
ims.cusip,
ims.shortname,
ims.defaultticker,
ims.coupon,
ims.maturity,
ims.SNPRatingdisp,

(case when ims.maturity <= add_months(mtd.currdate,12) then 'Short'
      when ims.maturity > add_months(mtd.currdate,12) and ims.maturity <= add_months(mtd.currdate,33) then '2 YR'
      when ims.maturity > add_months(mtd.currdate,33) and ims.maturity <= add_months(mtd.currdate,45) then '3 YR'
      when ims.maturity > add_months(mtd.currdate,45) and ims.maturity <= add_months(mtd.currdate,81) then '5 YR'
      when ims.maturity > add_months(mtd.currdate,81) and ims.maturity <= add_months(mtd.currdate,189) then '10 YR'
      when ims.maturity > add_months(mtd.currdate,189)  then '30 YR'      
      else 'NA' 
end) Maturity_Bucket,
ims.trading_sector,
mtd.currdate,
mtd.curavgprice,
mtd.curavgspread,
mtd.curtotalvolume,
mtd.curtotaltrades,
mtd.curavgyield - (mtd.curavgspread/100) as bench_yield,
median(inv.bid_level) as med_bid_inv,
median(inv.offer_level) as med_off_inv,
(idc.yield - (mtd.curavgyield - (mtd.curavgspread/100)))*100 as impl_bid_idc_spr,
idc.bid_price as bid_idc


from marketdata_currenttradesummary mtd,
instrument_mastersecurity ims,
marketdata_mainventory inv,
marketdata_idcpricing idc

where mtd.issueid = ims.id
and mtd.currdate = inv.currdate
and ims.id = inv.issueid
and ims.id = idc.issueid
and mtd.currdate = idc.currdate
and mtd.sizebucket = 1 

and ims.cusip in (select cusip from CUSIPS_PRICING)
and mtd.currdate between to_date('11/01/2013','MM/DD/YYYY') and to_date('12/31/2013','MM/DD/YYYY')
and mtd.sprdoryld = 0
AND reporting_side = 'A'
group by 
ims.cusip,
ims.shortname,
ims.defaultticker,
ims.coupon,
ims.maturity,
ims.SNPRatingdisp,
(case when ims.maturity <= add_months(mtd.currdate,12) then 'Short'
      when ims.maturity > add_months(mtd.currdate,12) and ims.maturity <= add_months(mtd.currdate,33) then '2 YR'
      when ims.maturity > add_months(mtd.currdate,33) and ims.maturity <= add_months(mtd.currdate,45) then '3 YR'
      when ims.maturity > add_months(mtd.currdate,45) and ims.maturity <= add_months(mtd.currdate,81) then '5 YR'
      when ims.maturity > add_months(mtd.currdate,81) and ims.maturity <= add_months(mtd.currdate,189) then '10 YR'
      when ims.maturity > add_months(mtd.currdate,189)  then '30 YR'      
      else 'NA' 
end),mtd.currdate,
mtd.curavgprice,
mtd.curavgspread,
mtd.curtotalvolume,
mtd.curtotaltrades,
ims.trading_sector,
idc.tsyspread ,
idc.bid_price,
mtd.curavgyield - (mtd.curavgspread/100),
(idc.yield - (mtd.curavgyield - (mtd.curavgspread/100)))*100   
   
    """ 
import pandas as pd
import numpy as np
import pandas.io.sql as psql
import pandasql as pysqldf
import pandas.io.sql as psql
import cx_Oracle
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from sklearn import metrics
from datetime import datetime, timedelta
from pandas.io.data import DataReader
from datetime import datetime


#trades = psql.read_frame(sql, con)  
    
trades = pd.read_csv('C:/Users/asedgwick/Desktop/ML Bond Pricing/test_data.csv')    
trades = trades.dropna()    
    
#colsolidate a few of the sectors
trades['TRADING_SECTOR'][trades['TRADING_SECTOR'].isin(['Asia IG','Unassigned','Latam Corps'])]='Other'

trades['INDUSTRY'] = 'Industrial'
trades['INDUSTRY'][trades['TRADING_SECTOR'].isin(['Banks/Finance','Insurance',''])]='Finance'
trades['INDUSTRY'][trades['TRADING_SECTOR'].isin(['Utility'])]='Utility'

#Drop Supranationals
trades = trades[trades['TRADING_SECTOR']!='Supranational']

#Create Dummy Variables for Maturity and sector

trades = trades.join(pd.get_dummies(trades['MATURITY_BUCKET'], prefix='MAT'))
trades = trades.drop(['MATURITY_BUCKET'], axis=1)
trades = trades.join(pd.get_dummies(trades['TRADING_SECTOR'], prefix='SEC'))
trades = trades.drop(['TRADING_SECTOR'], axis=1)
trades = trades.join(pd.get_dummies(trades['INDUSTRY'], prefix='Ind'))
trades = trades.drop(['INDUSTRY'], axis=1)

#Bucket the Ratings Dimport pandas as pd

#AAA	1
#AA+	2
#AA	3
#AA-	4
#A+	5
#A	6
#A-	7
#BBB+	8
#BBB	9
#BBB-	10
#BB+	11
#BB	12
#BB-	13
#B+	14
#B	15
#B-	16
#CCC+	17
#CCC	18
#CCC-	19
#CC	20
#C	21
#D	25
#A-1	32
#NA	98
#NR	99

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

trades['CURRDATE'] = pd.to_datetime(trades['CURRDATE'])    

trades = trades.drop(['SHORTNAME','MATURITY','CURAVGPRICE','CURTOTALTRADES','CURTOTALVOLUME'],axis=1)    

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
data = data.dropna()

# Filter the data
data = data[data['CURAVGSPREAD']>=0]
data = data[data['IMPL_BID_IDC_SPR']>=0]
data = data[data['CURAVGSPREAD']<=750]
data = data[data['IMPL_BID_IDC_SPR']<=750]
data = data[(data['MED_BID_INV']-data['MED_OFF_INV']<100) & (data['MED_BID_INV']-data['MED_OFF_INV']>-100)]

data = data.join(pd.get_dummies(data['DEFAULTTICKER']))
data = data.drop(['DEFAULTTICKER'], axis=1)


####Ridge Regression 

from sklearn import linear_model, metrics, tree, cross_validation, ensemble
import matplotlib.pyplot as plt
import statsmodels.api as sm

#X = data.drop(['Date','SHORTNAME','CUSIP', 'DEFAULTTICKER','CURAVGSPREAD','MATURITY','CURRDATE','CURAVGPRICE','CURTOTALVOLUME','CURTOTALTRADES','BENCH_YIELD'],axis=1)
#y = data['CURAVGSPREAD']
#
#
#lm = linear_model.Ridge()
#lm.fit(X,y)
#print 'R-sq:',metrics.r2_score(lm.predict(X),y)
#print 'MSE:',metrics.mean_squared_error(lm.predict(X),y)
#
# 
#lm_OLS = sm.OLS(y,X)   
#res = lm_OLS.fit()
#res.summary()


'add duration
'add bond age


X = data[['DEFAULTTICKER','IMPL_BID_IDC_SPR','MED_BID_INV','MED_OFF_INV','Close_LQD','Rat_A','Rat_AA','Rat_AAA','Rat_B','Rat_BB','Rat_BBB','Ind_Finance','Ind_Industrial','MAT_10 YR', 'MAT_2 YR', 'MAT_3 YR', 'MAT_30 YR', 'MAT_5 YR']]
X = X.join(pd.get_dummies(X['DEFAULTTICKER']))
X = X.drop(['DEFAULTTICKER'], axis=1)
y = data['CURAVGSPREAD']
lm = linear_model.Ridge()
lm.fit(X,y)
print 'R-sq:',metrics.r2_score(lm.predict(X),y)
print 'MSE:',metrics.mean_squared_error(lm.predict(X),y)

#X = data['IMPL_BID_IDC_SPR']
#y = data['CURAVGSPREAD']
#lm_OLS = sm.OLS(y,X)   
#res = lm_OLS.fit()
#res.summary()


plt.scatter(data.CURAVGSPREAD,lm.predict(X), cmap=plt.cm.jet,c='blue')
xlabel('TRACE') 
ylabel('Linear Prediction') 
plt.show()


#pd.tools.plotting.scatter_matrix(data[['CURAVGSPREAD','IMPL_BID_IDC_SPR','MED_BID_INV','MED_OFF_INV','Close_LQD']], alpha=0.2, diagonal='hist')
#plt.show
x_train, x_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=.3)

rf = RandomForestRegressor()
rf.fit(x_train,y_train)

print 'Train R-sq:',metrics.r2_score(rf.predict(x_train),y_train)
print 'Train MSE:',metrics.mean_squared_error(rf.predict(x_train),y_train)
print 'Test R-sq:',metrics.r2_score(rf.predict(x_test),y_test)
print 'Test MSE:',metrics.mean_squared_error(rf.predict(x_test),y_test)
print 'IDC MSE:',metrics.mean_squared_error(data['IMPL_BID_IDC_SPR'], data['CURAVGSPREAD'])

plt.scatter(data.CURAVGSPREAD,rf.predict(X), cmap=plt.cm.jet,c='blue')
xlabel('TRACE') 
ylabel('Random Forest Prediction') 
plt.show()

plt.scatter(data.CURAVGSPREAD,data.IMPL_BID_IDC_SPR, cmap=plt.cm.jet,c='blue')
xlabel('TRACE') 
ylabel('IDC') 
plt.show()


#output = pd.DataFrame({'Observed':y_test, 'Predict': rf.predict(x_test)})
#output.to_csv('C:/ML_Test.csv')

#Test MSE of 150
#IDC MSE of 466

#from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingRegressor
gb = GradientBoostingRegressor()
gb.fit(x_train,y_train)

print 'Train R-sq:',metrics.r2_score(gb.predict(x_train),y_train)
print 'Train MSE:',metrics.mean_squared_error(gb.predict(x_train),y_train)
print 'Test R-sq:',metrics.r2_score(gb.predict(x_test),y_test)
print 'Test MSE:',metrics.mean_squared_error(gb.predict(x_test),y_test)








