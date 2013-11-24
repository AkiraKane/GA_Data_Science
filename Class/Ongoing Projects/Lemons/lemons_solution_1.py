"""
This script builds a mode to determine if
a Car is a good buy (0) or bad buy (1)
"""
 
### IMPORT MODULES
import numpy as np
import pandas as pd
from sklearn import metrics, naive_bayes, cross_validation, feature_selection
 
 
def main():
    ### IMPORT DATA SETS
    l_train = pd.read_csv('lemon_training.csv')
    l_test = pd.read_csv('lemon_test.csv')
 
 
    ### SET UP FEATURES
    # make dummies
    make_train = pd.get_dummies(l_train['Make'])
    make_test = pd.get_dummies(l_test['Make'])
    l_train = l_train.join(make_train[['CHEVROLET', 'DODGE']])
    l_test = l_test.join(make_test[['CHEVROLET', 'DODGE']])
 
    # wheel type dummies
    wheel_train = pd.get_dummies(l_train['WheelType'])
    wheel_test = pd.get_dummies(l_test['WheelType'])
    l_train = l_train.join(wheel_train)
    l_test = l_test.join(wheel_test)
 
    # size dummies
    size_train = pd.get_dummies(l_train['Size'])
    del size_train['SMALL SUV']
    del size_train['SMALL TRUCK']
    del size_train['VAN']
    size_test = pd.get_dummies(l_test['Size'])
    del size_test['SMALL SUV']
    del size_test['SMALL TRUCK']
    del size_test['VAN']
    l_train = l_train.join(size_train)
    l_test = l_test.join(size_test)
 
    # big 3 dummies
    big3_train = pd.get_dummies(l_train['TopThreeAmericanName'])
    big3_test = pd.get_dummies(l_test['TopThreeAmericanName'])
    big3_train = big3_train.rename(columns={'OTHER': 'OTHER_big3', 'CHRYSLER': 'CHRYSLER_big3', 'FORD': 'FORD_big3'})
    big3_test = big3_test.rename(columns={'OTHER': 'OTHER_big3', 'CHRYSLER': 'CHRYSLER_big3', 'FORD': 'FORD_big3'})
    l_train = l_train.join(big3_train)
    l_test = l_test.join(big3_test)
 
    # fill in empty benchmark prices
    l_train['MMRAcquisitionAuctionAveragePrice'] = l_train['MMRAcquisitionAuctionAveragePrice'].fillna(np.mean(l_train['MMRAcquisitionAuctionAveragePrice']))
    l_train['MMRAcquisitionAuctionCleanPrice'] = l_train['MMRAcquisitionAuctionCleanPrice'].fillna(np.mean(l_train['MMRAcquisitionAuctionCleanPrice']))
    l_train['MMRCurrentAuctionAveragePrice'] = l_train['MMRCurrentAuctionAveragePrice'].fillna(np.mean(l_train['MMRCurrentAuctionAveragePrice']))
    l_train['MMRCurrentAuctionCleanPrice'] = l_train['MMRCurrentAuctionCleanPrice'].fillna(np.mean(l_train['MMRCurrentAuctionCleanPrice']))
    l_train['MMRCurrentRetailAveragePrice'] = l_train['MMRCurrentRetailAveragePrice'].fillna(np.mean(l_train['MMRCurrentRetailAveragePrice']))
    l_train['MMRCurrentRetailCleanPrice'] = l_train['MMRCurrentRetailCleanPrice'].fillna(np.mean(l_train['MMRCurrentRetailCleanPrice']))
    l_test['MMRAcquisitionAuctionAveragePrice'] = l_test['MMRAcquisitionAuctionAveragePrice'].fillna(np.mean(l_test['MMRAcquisitionAuctionAveragePrice']))
    l_test['MMRAcquisitionAuctionCleanPrice'] = l_test['MMRAcquisitionAuctionCleanPrice'].fillna(np.mean(l_test['MMRAcquisitionAuctionCleanPrice']))
    l_test['MMRCurrentAuctionAveragePrice'] = l_test['MMRCurrentAuctionAveragePrice'].fillna(np.mean(l_test['MMRCurrentAuctionAveragePrice']))
    l_test['MMRCurrentAuctionCleanPrice'] = l_test['MMRCurrentAuctionCleanPrice'].fillna(np.mean(l_test['MMRCurrentAuctionCleanPrice']))
    l_test['MMRCurrentRetailAveragePrice'] = l_test['MMRCurrentRetailAveragePrice'].fillna(np.mean(l_test['MMRCurrentRetailAveragePrice']))
    l_test['MMRCurrentRetailCleanPrice'] = l_test['MMRCurrentRetailCleanPrice'].fillna(np.mean(l_test['MMRCurrentRetailCleanPrice']))
 
    # state dummies
    st_train = pd.get_dummies(l_train['VNST'])
    st_test = pd.get_dummies(l_test['VNST'])
    l_train = l_train.join(st_train[['FL', 'KY', 'MD', 'TX']])
    l_test = l_test.join(st_test[['FL', 'KY', 'MD', 'TX']])
 
    # create new variables from others
    l_train['VehicleAge sqr'] = l_train['VehicleAge'] **2
    l_train['VehYear log'] = np.log(l_train['VehYear'])
    l_train['VehOdo sqr'] = l_train['VehOdo'] **2
    l_train['VehOdo log'] = np.log(l_train['VehOdo'])
    l_test['VehicleAge sqr'] = l_test['VehicleAge'] **2
    l_test['VehYear log'] = np.log(l_test['VehYear'])
    l_test['VehOdo sqr'] = l_test['VehOdo'] **2
    l_test['VehOdo log'] = np.log(l_test['VehOdo'])
 
    l_train = l_train.dropna(axis=1)
    l_test = l_test.dropna(axis=1)
    features = list(l_train.describe().columns)
    features.remove('RefId')
    features.remove('IsBadBuy')
    features.remove('IsOnlineSale')
 
 
    ### CREATE TEST AND TRAINING SETS
    train_features = l_train[features].values
    train_class = l_train.IsBadBuy.values
    # Seed PRNG
    np.random.seed(1234)
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(train_features, train_class, test_size=.35)
 
 
    ### BUILD MODEL
    model = naive_bayes.GaussianNB().fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
 
 
    ### STATS
    print 'training:\n', metrics.confusion_matrix(y_train, y_pred_train)
    print metrics.classification_report(y_train, y_pred_train)
    print 'test:\n', metrics.confusion_matrix(y_test, y_pred_test)
    print metrics.classification_report(y_test, y_pred_test)
    fpr_train, tpr_train, thresholds_train = metrics.roc_curve(y_train, y_pred_train, pos_label=1)
    fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y_test, y_pred_test, pos_label=1)
    print 'train MA: ', model.score(X_train, y_train)
    print 'test MA: ', model.score(X_test, y_test)
    print 'train AUC: ', metrics.auc(fpr_train, tpr_train)
    print 'test AUC: ', metrics.auc(fpr_test, tpr_test)
 
    # Cross Validation
    AUCs = []
    for i in xrange(20):
        X_train, X_test, y_train, y_test = \
            cross_validation.train_test_split(train_features, train_class, test_size=.35)
        y_pred_test = model.fit(X_train, y_train).predict(X_test)
        fpr_test, tpr_test, thresholds_test = metrics.roc_curve(y_test, y_pred_test, pos_label=1)
        AUCs.append(metrics.auc(fpr_test, tpr_test))
    print 'avgerage test AUC cross val: ', np.mean(AUCs)
    print 'min test AUC cross val: ', min(AUCs)
 
 
    ### OUTPUT PREDICTIONS FOR OSS DATA
    OSS_features = l_test[features].values
    y_pred_OSS = model.predict(OSS_features)
    submission = pd.DataFrame({ 'RefId' : l_test.RefId, 'prediction' : y_pred_OSS })
    submission.to_csv('submission_NB.csv')
 
 
if __name__ == '__main__':
    main()