#!/usr/bin/env python
 
'''
Joe Carli
Data Science, Ongoing Homework 01, Submission 02
October 12, 2013
 
Develop a model to predict 2011 salaries from 2011 stats.
 
Then test the model with 2012 data and try to beat the benchmark.
Benchmark MSE: 2.094098e+13
R-Squared: 0.1322
 
This model uses both statistics (AB's, RBI's, etc.) in addition to
the team someone plays for, what hand they throw with, and what hand
they bat with.
 
Including bats, throws, and team, the possible feature space included
23 components. A brute force approach of 23 choose k features, from k=1
to k=23, was taken to find the highest-scoring model. Other features may
be considered in the future, including whether a player is a pitcher
(perhaps if 0 <= AB < G).
 
The chosen feature set and score from training data:
Best score (index 8388606): 0.33827180, MSE 1.39444801e+13
['G', 'AB', 'R', 'H', 'X2B', 'X3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 
'IBB', 'HBP', 'SH', 'SF', 'GIDP', 'height', 'weight', 'birthYear', '
bats_a', 'bats_b', 'throws_a', 'throws_b', '
team_a', 'team_b', 'team_d', 'team_c', 'team_d', 'team_e']
 
Notes: 
 - bats_a and bats_b are vectorized values [0,1] or [1,0] for bats "L" or bats "R"
 - throws_a and throws_b are similary vectorized
 - team_a,b,c,d,e are [0,0,0,0,1 - 1,1,1,1,0] representing the 30 Major League teams
 
Outputs submission.csv (yearID will always be 2012):
playerID, yearID, salary, predicted_salary
'''
 
import getopt
import sys
import numpy as np
import pandas as pd
from sklearn import linear_model, metrics 
 
# Global _teams dictionary
# skipped 0,0,0,0,0
_teams = {'ARI':pd.Series([0,0,0,0,1]),
        'ATL':pd.Series([0,0,0,1,0]),
        'BAL':pd.Series([0,0,0,1,1]),
        'BOS':pd.Series([0,0,1,0,0]),
        'CHA':pd.Series([0,0,1,0,1]),
        'CHN':pd.Series([0,0,1,1,0]),
        'CIN':pd.Series([0,0,1,1,1]),
        'CLE':pd.Series([0,1,0,0,0]),
        'COL':pd.Series([0,1,0,0,1]),
        'DET':pd.Series([0,1,0,1,0]),
        'FLO':pd.Series([0,1,0,1,1]),
        'HOU':pd.Series([0,1,1,0,0]),
        'KCA':pd.Series([0,1,1,0,1]),
        'LAA':pd.Series([0,1,1,1,0]),
        'LAN':pd.Series([0,1,1,1,1]),
        'MIL':pd.Series([1,0,0,0,0]),
        'MIA':pd.Series([0,0,0,0,0]),
        'MIN':pd.Series([1,0,0,0,1]),
        'NYA':pd.Series([1,0,0,1,0]),
        'NYN':pd.Series([1,0,0,1,1]),
        'OAK':pd.Series([1,0,1,0,0]),
        'PHI':pd.Series([1,0,1,0,1]),
        'PIT':pd.Series([1,0,1,1,0]),
        'SDN':pd.Series([1,0,1,1,1]),
        'SEA':pd.Series([1,1,0,0,0]),
        'SFN':pd.Series([1,1,0,0,1]),
        'SLN':pd.Series([1,1,0,1,0]),
        'TBA':pd.Series([1,1,0,1,1]),
        'TEX':pd.Series([1,1,1,0,0]),
        'TOR':pd.Series([1,1,1,0,1]),
        'WAS':pd.Series([1,1,1,1,0])}
 
def vectorizeHandedness(x):
    '''
    Return [0,1] for left-handed, [1,0] for right-handed.
    Returns a pandas Series object.
    '''
    if x == 'L':
        return pd.Series([0,1])
    else:
        return pd.Series([1,0])
# end of vectorizeHandedness()
 
def vectorizeLeague(x):
    '''
    Return [0,1] for the National League, [1,0] for the American League.
    Reutrns a pandas Series object.
    '''
    if x == 'NL':
        return pd.Series([0,1])
    else:
        return pd.Series([1,0])
# end of vectorizeLeague()
 
def vectorizeTeam(x):
    '''
    Looks up a team ID (3 letters) in the global _teams dictionary.
    Returns a 5-integer list of 0's and 1's as a vectorized
    representation of the team. Returns a pandas Series object.
    '''
    global _teams
    return _teams[x]
# end of vectorizeTeam()
 
def choose_iter(elements, length):
    '''
    Iterator code found on Stack Overflow.
    '''
    for i in xrange(len(elements)):
        if length == 1:
            yield (elements[i],)
        else:
            for next in choose_iter(elements[i+1:len(elements)], length-1):
                yield (elements[i],) + next
def choose(l, k):
    '''
    n choose k code found on Stack Overflow.
    Uses choose_iter as its helper routine.
    '''
    return list(choose_iter(l, k))
# end of choose()
 
def genFeatureSets(possibleFeatures,k):
    fsList = choose(possibleFeatures,k)
    fsVectorizedList = list()
    for fs in fsList:
        vecList = list()
        for i in range(len(fs)):
            if fs[i]=='bats':
                vecList.extend(['bats_a','bats_b'])
            elif fs[i]=='throws':
                vecList.extend(['throws_a','throws_b'])
            elif fs[i]=='teamID':
                vecList.extend(['team_a','team_b','team_d','team_c',
                    'team_d','team_e'])
            else:
                vecList.append(fs[i])
        fsVectorizedList.append(vecList)
    return fsVectorizedList
# end of genFeatureSets()
 
def main(trainfile,testfile,doBruteForce=False,fs=[]):
    bb2011 = pd.read_csv(trainfile)
    bb2012 = pd.read_csv(testfile)
    
    # Benchmark feature set
    #fs = ['G', 'AB', 'R', 'H', 'X2B', 'X3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF']
    # Note: G_batting missing from 2012 data
    possibleFeatures = ['G', 'AB', 'R', 'H', \
        'X2B', 'X3B', 'HR', 'RBI', 'SB', \
        'CS', 'BB', 'SO', 'IBB', 'HBP', \
        'SH', 'SF', 'GIDP', 'height', 'weight', \
        'birthYear','bats','throws','teamID']
 
    # vectorize bats, throws, and teamID so they can be used by the linear model
    bb2011[['bats_a','bats_b']] = bb2011['bats'].apply(vectorizeHandedness)
    bb2011[['throws_a','throws_b']] = bb2011['throws'].apply(vectorizeHandedness)
    bb2011[['team_a','team_b','team_c','team_d','team_e']] = \
        bb2011['teamID'].apply(vectorizeTeam)
 
    bb2012[['bats_a','bats_b']] = bb2012['bats'].apply(vectorizeHandedness)
    bb2012[['throws_a','throws_b']] = bb2012['throws'].apply(vectorizeHandedness)
    bb2012[['team_a','team_b','team_c','team_d','team_e']] = \
        bb2012['teamID'].apply(vectorizeTeam)
 
    # Extract the training response for future use (salary)
    train_y = bb2011['salary'].values
    min2011_salary = np.min(train_y)
 
    # If asked to brute force, compute the list of all possible feature sets
    if doBruteForce==True:
        fsList = list()
        numPossibleFeatures = len(possibleFeatures)
        for k in range(1,numPossibleFeatures+1):
            print "Generating %d choose %d feature sets ..."%(numPossibleFeatures,k)
            fsList_k = genFeatureSets(possibleFeatures,k)
            fsList.extend(fsList_k)
 
        # Fit and predict for each feature set, tracking the best results
        bestScore = 0.0
        bestScoreIndex = 0
        lowestMSE = 0.0
        lowestMSEIndex = 0
        scoreList = list()
        mseList = list()
        fsListLen = len(fsList)
        print "%d feature sets will be evaluated"%(fsListLen) 
        for i in range(len(fsList)):
            train_X = bb2011[fsList[i]].values
            lm = linear_model.Ridge()
            lm.fit(train_X,train_y)
            thisPred = lm.predict(train_X)
            thisScore = lm.score(train_X, train_y)
            thisMSE = metrics.mean_squared_error(thisPred, train_y)
            scoreList.append(thisScore)
            mseList.append(thisMSE)
            # update highest score (or set to first calculated result)
            if thisScore > bestScore or i==0:
                bestScore = thisScore
                bestScoreIndex = i
            # update lowestMSE (or set to first calculated result)
            if thisMSE < lowestMSE or i==0:
                lowestMSE = thisMSE
                lowestMSEIndex = i
            if (i%500000) == 0:
                print "completed index %d of %d"%(i,fsListLen-1)
 
        # Note: Have confirmed that highest score and lowest MSE came from the same model
        # print the feature set chosen by best score and lowest MSE
        print ("Best score (index %d): %0.8f, MSE %0.8e"%
            (bestScoreIndex,bestScore,mseList[bestScoreIndex]))
        print fsList[bestScoreIndex]
        print ""
        print ("Lowest MSE (index %d): %0.8e, score %0.8f"%
            (lowestMSEIndex,lowestMSE,scoreList[lowestMSEIndex]))
        print fsList[lowestMSEIndex]
 
        # If we did brute force, set fs to the best feature set
        fs = fsList[bestScoreIndex]
 
    # Extract the 2011 training data (train_y was pulled out above)
    train_X = bb2011[fs].values
 
    # Prepare the 2012 test data 
    test_X = bb2012[fs].values
    test_y = bb2012['salary'].values
    bb2012_csv = bb2012[['playerID','yearID','salary']]
 
    # Train the model against 2011 data
    lm = linear_model.Ridge()
    lm.fit(train_X, train_y)
    pred_y = lm.predict(train_X)
    trainScore = lm.score(train_X, train_y)
 
    # Post-process predictions of the test data!
    # For any salary too low, assume our model choked and set to 2011 min 
    # Granted, 2012 minimum is higher, but this is a best effort
    for i in range(len(pred_y)):
        if pred_y[i] < min2011_salary:
            pred_y[i] = min2011_salary
 
    print 'R-squared on train_X:',trainScore
    print 'MSE on training data:',metrics.mean_squared_error(pred_y, train_y)
 
    # Predict 2012 salaries from 2012 stats (still weird)
    pred_2012_salaries = lm.predict(test_X)
 
    # Add the 2012 salary predictions to the limited 2012 DataFrame
    bb2012_csv['predicted_salary'] = pred_2012_salaries
    bb2012_csv.to_csv('submission.csv',index=False)
 
    # Print the score for our 2012 test for sanity's sake
    print 'R-squared on test_X:',lm.score(test_X, test_y)
    print 'MSE on testing data:',metrics.mean_squared_error(lm.predict(test_X), test_y)
 
# end of main()
 
if __name__=="__main__":
    # What the brute force method told us to use:
    fs=['G', 'AB', 'R', 'H', 'X2B', 'X3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF', 'GIDP', 'height', 'weight', 'birthYear', 'bats_a', 'bats_b', 'throws_a', 'throws_b', 'team_a', 'team_b', 'team_d', 'team_c', 'team_d', 'team_e']
 
    doBruteForce = False
    try:
        opts, args = getopt.getopt(sys.argv[1:],'b')
        for opt, arg in opts:
            if opt == '-b':
                doBruteForce = True
    except:
        print "usage: %s [OPTIONS]"%(sys.argv[0])
        print "Options:"
        print "\t-b     Perform brute force of all 23 choose k feature sets"
        print "\t           (from k=1 to k=23)"
        print ""
        print ("Writes submission.csv using its default feature set or\n"
            "whatever feature set is found to have the highest score using\n"
            "the brute force technique.\n")
        sys.exit(1)
 
    if doBruteForce == False:
        print "Using this feature set:"
        print fs
 
    main('baseball_training_2011.csv','baseball_test_2012.csv',doBruteForce,fs)
 
# end of baseball.py