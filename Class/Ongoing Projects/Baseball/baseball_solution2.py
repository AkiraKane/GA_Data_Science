## Baseball players salary prediction
"""
Problem:  Predict baseball player salaries for 2012
Data:       Training data with text that contains salaries for multiple years
Model:      Salary(2012) = Max( Ridge{Salary(2011); Player_Stats(2011); Age; Max(Age - 31, 0)}, 
                                Min_contract_salary(2012))
 
"""
 
### import modules
import os
os.system('clear')
 
print 'importing libraries ...'
import pdb
import pandas as pd
from sklearn import linear_model, metrics, feature_selection
import matplotlib.pyplot as plt
import copy
 
### load data
print 'loading data ...'
pd.set_option("display.max_info_columns", 100)
pd.set_option("display.max_columns", 100)
bball = pd.read_csv('baseball.csv')
 
data_descr = {
    'yearID':   'data year',
    
    'bats':     'hitter type , Right(R), Left(L) or Both(B) ?',
    'throws':   'thrower type, Right(R), Left(L) ?',
    
    'AB':           '# of at-bats',
    'BB':           '# of bases on ball or walks ?',
    'HR':           '# of home runs',
    'G':            '# of games played',
    'G_batting':    '# of games batted ?',
    'GIDP':         '# of ground into double play',
    'H':            '# of hits',
    'HBP':          '# of hits by pitch',
    'IBB':          '# of intentional bases on ball',
    'R':            '# of runs',
    'RBI':          '# of runs batted in',
    'SF':           '# of sacrifice flies',
    'SH':           '# of sacrifice hits',
    'SO':           '# of strike outs',
    'X2B':          '# of ???'
}
 
b2011 = pd.read_csv('baseball_training_2011.csv')
b2012 = pd.read_csv('baseball_test_2012.csv')
 
### inspect data
 
#   Drop these cols since bball.shape has 59 cols
#   Adding a col does not display data for the columns even though
#       I am setting max_info_columns to 100 
del bball['nameNote']
del b2011['nameNote']
del bball['deathCity']
del b2011['deathCity']
del bball['deathState']
del b2011['deathState']
 
#   Fix playerIDs for some rows
bball.loc[778:786,'playerID'] = 'baezjo01'
b2012.loc[778:786,'playerID'] = 'baezjo01'
b2011.loc[778:786,'playerID'] = 'baezjo01'
 
### transform data
 
b2010 = bball[bball['yearID'] == 2010]
 
#   Calculate Age
b2010['Age'] = b2010['yearID'] - b2010['birthYear']
b2011['Age'] = b2011['yearID'] - b2011['birthYear']
b2012['Age'] = b2012['yearID'] - b2012['birthYear']
 
#       {salary - coef_ * salary.pyear} vs. Age scatter plot
#           displays different slopes at age ~ 31 for 2011 data
#           should I create different age inflections for 2010, 11 & 12 data ?
#       salary increases by $51K per year thru 31 & 
#            then declines $226K per year for every year over 31
def age_h31(age):
    if age <= 31:
        return 0
    else:
        return age - 31
 
b2010['Age_H31'] = b2010['Age'].apply(age_h31)
b2011['Age_H31'] = b2011['Age'].apply(age_h31)
b2012['Age_H31'] = b2012['Age'].apply(age_h31)
 
#   Merge previous year's salary as 'salary.pyear' & other player stats
#   Drop SH.pyear from model since p-value is 0.5644
#   Why does G.pyear & AB.pyear & R.pyear have a -ve coeff in the kitchen-sink model ?
#   Add salary.pyear, RBI.pyear, since they have the highest corr with salary
#       Skip HR.pyear & H.pyear & SF.pyear since the correlation with RBI.pyear is high
#           RBI.pyear has a higher correlation with salary than HR.pyear
#       Skip BB.pyear since the correlation with IBB.pyear is high
#       Skip AB.pyear & X2B.pyear & SO.pyear & G.pyear since the correlation with R.pyear is high
vars_regr = [
            #'G', 
            #'AB',
            'R', 
            #'H', 
            #'X2B',
            #'X3B', 
            #'HR', 
            'RBI',
            'SB', 
            #'CS', 
            #'BB',
            #'SO',
            'IBB',
            'HBP', 
            #'SH', 
            #'SF',
            'salary']
            
vars_merge = copy.copy(vars_regr)
vars_merge.append('playerID')
tmp_b2010 = b2010[vars_merge]
tmp_b2011 = b2011[vars_merge]
 
b2011p = pd.merge(b2011, tmp_b2010, how='left', on='playerID', suffixes=('', '.pyear'))
b2012p = pd.merge(b2012, tmp_b2011, how='left', on='playerID', suffixes=('', '.pyear'))
 
#   Fill salary.pyear NaNs with that year's min salary
min_sal_2010 = min(b2010['salary'])
min_sal_2011 = min(b2011['salary'])
min_sal_2012 = min(b2012['salary'])
b2011p['salary.pyear'].fillna(min_sal_2010, inplace=True)
b2012p['salary.pyear'].fillna(min_sal_2011, inplace=True) 
 
### build training & test data
                        
vars_X = [var + '.pyear' for var in vars_regr]
vars_X.append('Age')
vars_X.append('Age_H31')
vars_XyID = copy.copy(vars_X)
vars_XyID.append('salary')
vars_XyID.append('playerID')
 
# drop NaNs only on vars of interest
b2011pn = b2011p[vars_XyID].dropna()
train_X = b2011pn[vars_X].values
train_y = b2011pn['salary'].values
 
# fill NaNs for test data - all numerics except salary.pyear
#   they did not play in prev year
b2012pf = b2012p[vars_XyID].fillna(0)
print "b2012pf corr:"
print b2012pf.corr()
test_X = b2012pf[vars_X].values
test_y = b2012pf['salary'].values
 
feat_p_vals = feature_selection.f_regression(train_X, train_y)[1]
print " feature p-values:{0}".format(zip(vars_X, feat_p_vals))
max_p_val = max(feat_p_vals)
max_p_value_pos = feat_p_vals.tolist().index(max_p_val)
print " max p-value:{0} for {1}".format(feat_p_vals[max_p_value_pos], vars_X[max_p_value_pos])  
 
### build prediction model
print 'building prediction model ...'
lm = linear_model.Ridge()
lm.fit(train_X, train_y)
 
### test model results
def adj_r_sq (model, xvals, yvals):
    adj = 1 - float(len(yvals)-1)/(len(yvals)-len(model.coef_)-1)*(1 - model.score(xvals,yvals))
    return adj
 
# Checking performance
print 'R-Squared:',lm.score(train_X, train_y)
print 'Adj-R-sq:', adj_r_sq(lm, train_X, train_y)
# Checking MSE
print 'MSE:',metrics.mean_squared_error(lm.predict(train_X), train_y)
 
# Display coeffs
print "Ridge coef_:{0}".format(zip(vars_X, lm.coef_))
 
 
predict_y = lm.predict(test_X)
 
#   Check for -ve values
neg_predict_y_pos = [i for i,y in enumerate(predict_y) if y < 0]
if len(neg_predict_y_pos) > 0:
    print "Negative salary predictions for:" 
    
for i in range(len(neg_predict_y_pos)):
    print " playerID = {0}, nameLast = {1}, salary = {2}, Age = {3}, Prediction = {4}".format(
        b2012.ix[i,'playerID'], b2012.ix[i,'nameLast'], b2012.ix[i, 'Age'], b2012.ix[i,'salary'], 
        predict_y[neg_predict_y_pos[i]])
 
# Plot predictions vs. actual to test for bias & variance
plt.scatter(test_y,predict_y)
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Predicting 2012 Salary")
plt.show()
 
### output results
submit_filename = 'sub_ts_key_num_cat.csv'
print "Outputting submission file as ", submit_filename
b2012_csv = b2012[['playerID','yearID', 'salary']]
 
# Replace predicted salaries that are less than min sal with min sal
minsal_predict_y = predict_y
minsal_predict_y[minsal_predict_y < min_sal_2012] = min_sal_2012
b2012_csv['predicted'] = predict_y
b2012_csv.to_csv(submit_filename)