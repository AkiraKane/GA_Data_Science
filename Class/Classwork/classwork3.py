# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 19:35:43 2013

@author: alexandersedgwick
"""

import numpy as numpy
import pandas as pd


def mVectorMatrix(vector,matrix):
    answer = list()   
    for count1 in range(len(matrix[0])):
        sum = 0  
        for count2 in range(len(vector)):
            #print (vector[count2]*matrix[count2][count1])
            sum = sum + (vector[count2]*matrix[count2][count1])
        answer.append(sum)
    return answer


def mMatrixMatrix(matrix1,matrix2):
    answer = list()
    for count3 in range(len(matrix1)):
        answer.append(mVectorMatrix(matrix1[count3],matrix2))
    return answer


def mIdentityMatrix(Dimension):   
    return [ [ 1 if col == row else 0 for col in range(Dimension) ] for row in range(Dimension) ]



a=[2, 4, 6, 8]
b=[[1, 2], [3, 4], [9, 6], [2, 8]]


mVectorMatrix(a,b)
dot(a,b)



a=[[1,2,3],[4,5,6]]
b=[[1,4],[2,5],[3,6]]

mMatrixMatrix(a,b)
dot(a,b)


mIdentityMatrix(10)
eye(10)

import timeit

a = numpy.eye(10)
%timeit a


import pandas as pd
from urllib import urlopen

df = pd.DataFrame()

for x in range(1,30):
    page = urlopen("http://stat.columbia.edu/~rachel/datasets/nyt"+str(x)+".csv")
    data = pd.read_csv(page)
    df = df.append(data)    
groups = df.groupby(['Age','Gender','Signed_In'])
ans = groups.Clicks.sum()/groups.Impressions.sum()
ans.to_csv('/users/alexandersedgwick/dropbox/development/ga/data/test_output_class.csv', header=True)






