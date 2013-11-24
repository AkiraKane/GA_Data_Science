import numpy as np
import scipy
import pandas as pd
import math
import re
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model, covariance, svm, metrics, feature_selection, ensemble
from sklearn.metrics import *
 
####Import the test and training datasets
cars_training = pd.io.parsers.read_csv('lemon_training.csv', delimiter=',')
cars_training = cars_training.dropna(subset=['WheelType'])
cars_training = cars_training.dropna(subset=['MMRCurrentAuctionCleanPrice'])
 
cars_test = pd.io.parsers.read_csv('lemon_test.csv', delimiter=',')
cars_test_backup = cars_test
cars_test = cars_test.fillna(value=0)
#cars_test= cars_test.dropna(subset=['Nationality'])
#cars_test = cars_test.dropna(subset=['MMRCurrentAuctionCleanPrice'])
 
####break into subclasses and make dummy variables
 
def suvMe(x):
    if re.search('SUV', str(x)) != None:
        return 1
    else:
        return 0
 
cars_training['suv'] = cars_training['SubModel'].apply(suvMe)
 
 
def minivanMe(x):
    if re.search('MINIVAN', str(x)) != None:
        return 1
    else:
        return 0
 
cars_training['minivan'] = cars_training['SubModel'].apply(minivanMe)
 
 
def sedanMe(x):
    if re.search('SEDAN', str(x)) != None:
        return 1
    else:
        return 0
 
cars_training['sedan'] = cars_training['SubModel'].apply(sedanMe)
 
 
def utilityMe(x):
    if re.search('UTILITY',str(x)) != None:
        return 1
    else:
        return 0
 
cars_training['utility'] = cars_training['SubModel'].apply(utilityMe)
 
 
def wagonMe(x):
    if re.search('WAGON', str(x)) != None:
        return 1
    else:
        return 0
 
cars_training['wagon'] = cars_training['SubModel'].apply(wagonMe)
 
def twoDoorMe(x):
    if re.search('2D', str(x)) != None:
        return 1
    else:
        return 0
 
cars_training['twoDoor'] = cars_training['SubModel'].apply(twoDoorMe)
 
def AmericanMe(x):
        if re.search('AMERICAN', str(x)) != None:
                     return 1
        else:
                     return 0
 
cars_training['american'] = cars_training['Nationality'].apply(AmericanMe)
 
#Split  into Test and Training Sets
n = math.floor(len(cars_training)*0.7)
randomVect = np.hstack((np.ones(n, dtype=np.bool), np.zeros(len(cars_training) - n, dtype=np.bool)))
np.random.shuffle(randomVect)
CarsTrain, CarsTest = cars_training[randomVect], cars_training[randomVect == False]
####select variables for model
indVariables_training = CarsTrain[['VehicleAge',                #0.03 From feature_importances_
                                    'VehOdo',                   #0.16
                    'MMRAcquisitionAuctionAveragePrice',    #0.12
                    'MMRAcquisitionAuctionCleanPrice',      #0.13
                    'MMRAcquisitionRetailAveragePrice',     #0.12
                                    'MMRAcquisitonRetailCleanPrice',        #0.12
                    'VehBCost',                 #0.15
                                    'IsOnlineSale',             #0.004
                                    'WarrantyCost',             #0.08
                                    'suv',                  #0.005
                                    'minivan',                  #0.004
                                    'sedan',                    #0.009
                                    'utility',                  #0.003
                                    'wagon',                    #0.004
                                    'twoDoor',                  #0.005
                                    'american']]                #0.008
 
 
depVariables_training = CarsTrain[['IsBadBuy']]
 
indVariables_test = CarsTest[['VehicleAge', 'VehOdo','MMRAcquisitionAuctionAveragePrice', 'MMRAcquisitionAuctionCleanPrice', 'MMRAcquisitionRetailAveragePrice', 'MMRAcquisitonRetailCleanPrice', 'VehBCost', 'IsOnlineSale', 'WarrantyCost', 'suv', 'minivan', 'sedan', 'utility', 'wagon', 'twoDoor','american']]
depVariables_test = CarsTest[['IsBadBuy']]
'''
####logistic regression
logm = linear_model.LogisticRegression()
logm.fit_transform(indVariables_training, depVariables_training)
#returns 2 numbers equalling 1.  need one.
probability = logm.predict_proba(indVariables_training)
prob2 = []
def singleColumn(probability, prob2):
    for i in range(len(probability)):
        prob2.append(probability[i][1])
 
 
singleColumn(probability, prob2)
#What percentage of lemons are there in the test set?
np.sum(CarsTrain['IsBadBuy']) #3280
len(CarsTrain['IsBadBuy'])    #34021
1365.00/14581 #.09641104023985185
 
#p=0.096 is what number?  First make a list of the 14581 probability values, then sort, then find the nth number where
#n =34021-3280 = 30741
 
prob3 = np.sort(prob2)
prob3[30741] #0.17626856208392727
#So, predict if the score is above 0.17627
prob4 = []
def myPredict(prob2, prob4):
    for i in range(len(prob2)):
         if prob2[i] > 0.17627:
         prob4.append(1)
     else:
         prob4.append(0)
 
myPredict(prob2, prob4)
#put it into the dataset
CarsTrain['predicted'] = prob4
 
 
#false positives
print("Number of mislabeled points : %d" % (CarsTrain['predicted'] != CarsTrain['IsBadBuy']).sum())#5371
#15.78 % false positive rate
 
#scoring
logm.score(indVariables_training, prob4) #0.8995
 
####now let's try it for the test set
probability = logm.predict_proba(indVariables_test)
prob2 = []
def singleColumn(probability, prob2):
    for i in range(len(probability)):
        prob2.append(probability[i][1])
 
 
singleColumn(probability, prob2)
 
prob4 = []
def myPredict(prob2, prob4):
    for i in range(len(prob2)):
         if prob2[i] > 0.17627:
             prob4.append(1)
     else:
         prob4.append(0)
 
myPredict(prob2, prob4)
#put it into the dataset
CarsTest['predicted'] = prob4
 
#false positives and negatives
print("Number of mislabeled points : %d" % (CarsTest['predicted'] != CarsTest['IsBadBuy']).sum())#2231
2231.0/len(CarsTest)
#0.15300733831698787 false positive rate
#If you didn't guess at all, false positive rate of .0936149.  Worse than guessing 0!
 
#scoring
logm.score(indVariables_test, prob4) #0.906453
'''
####Random Forest
forest = sklearn.ensemble.ExtraTreesClassifier(n_estimators = 5000, bootstrap=True, n_jobs=10, max_depth=None, min_samples_split = 1)
forest.fit(indVariables_training, depVariables_training)
forestPrediction = forest.predict_proba(indVariables_training)
 
forest.feature_importances_
'''
array([ 0.03780753,  0.16485448,  0.12880375,  0.13040341,  0.12641865,
        0.12631422,  0.15103475,  0.00420493,  0.08871761,  0.00536126,
        0.00383558,  0.00944388,  0.00366368,  0.0046814 ,  0.00569383,
        0.00876104])
'''
prob2 = []
def singleColumn(probability, prob2):
    for i in range(len(probability)):
        prob2.append(probability[i][1])
 
singleColumn(forestPrediction,prob2)
 
####score
#sklearn.metrics.roc_auc_score(depVariables_test, forest.predict(indVariables_test)) #error: AUC is defined for binary classification only???
 
 
####prep data for cars_test
cars_test['suv'] = cars_test['SubModel'].apply(suvMe)
cars_test['minivan'] = cars_test['SubModel'].apply(minivanMe)
cars_test['sedan'] = cars_test['SubModel'].apply(sedanMe)
cars_test['utility'] = cars_test['SubModel'].apply(utilityMe)
cars_test['wagon'] = cars_test['SubModel'].apply(wagonMe)
cars_test['twoDoor'] = cars_test['SubModel'].apply(twoDoorMe)
cars_test['american']=cars_test['Nationality'].apply(AmericanMe) #throws an error???
#cars_test['american'] = 0  #fuck it
 
####Define Ind. Variables
finalIndVariables = cars_test[['VehicleAge',                #0.03 From feature_importances_
                                    'VehOdo',                   #0.16
                    'MMRAcquisitionAuctionAveragePrice',    #0.12
                    'MMRAcquisitionAuctionCleanPrice',      #0.13
                    'MMRAcquisitionRetailAveragePrice',     #0.12
                                    'MMRAcquisitonRetailCleanPrice',        #0.12
                    'VehBCost',                 #0.15
                                    'IsOnlineSale',             #0.004
                                    'WarrantyCost',             #0.08
                                    'suv',                  #0.005
                                    'minivan',                  #0.004
                                    'sedan',                    #0.009
                                    'utility',                  #0.003
                                    'wagon',                    #0.004
                                    'twoDoor',                  #0.005
                                    'american']]
 
#####fit the model
finalProbabilities = forest.predict_proba(finalIndVariables)
prob2 = []
singleColumn(finalProbabilities,prob2)
#finalPrediction = forest.predict(finalIndVariables)
#np.sum(finalPrediction)/float(len(finalPrediction))
 
finalTable = pd.DataFrame(cars_test['RefId'])
finalTable['RefId'] = cars_test['RefId']
finalTable['prediction'] = prob2
 
finalTable.to_csv('predictionsWSS_extraTrees.csv', sep=";")