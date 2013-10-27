# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 18:26:03 2013

@author: alexandersedgwick
"""

import numpy as np

x,y = False, False

if x:
    print 'apple'
elif y:
    print 'banana'
else:
    print 'sandwich'
    
    
    
x = 0 

while True:
    print 'hello!'
    x += 1
    if x>= 3:
        break
    
# Iterating through a vector    
    
a = [10,10,25,35,45]

for value in a:
    if value >= 30:
        print value
        

vector = [1,2,3]

matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]

npMatrix = np.matrix('1 2 3; 4 5 6; 7 8 9')



# Python notation

#note diff 

vector*3

[x*3 for x in vector]



def vectorMultiply(vector, value):
    return [x*value for x in vector]
    
vectorMultiply([2,4,6],8)


def matrixTranspose(matrix):
     return [     [row[i] for row in matrix] for i in range(len(matrix)+1)]







# Exercise

def vectorMultiply(vector, value):
    return [x*value for x in vector]


def matrixTranspose(matrix):
     return [[row[i] for row in matrix] for i in range(len(matrix)+1)]
     


#Test 

def mVectorMatrix(vector,matrix):
    answer = list()   
    for count1 in range(len(matrix[0])):
        sum = 0  
        for count2 in range(len(vector)):
            #print (vector[count2]*matrix[count2][count1])
            sum = sum + (vector[count2]*matrix[count2][count1])
        answer.append(sum)
    return answer

mVectorMatrix(a,b)
mVectorMatrix([1,2,3],b)


a=[2, 4, 6, 8]
b=[[1, 2], [3, 4], [9, 6], [2, 8]]

mVectorMatrix(a,b)

#Check via NUMPY
A = np.matrix('2 4 6 8')
B = np.matrix([[1,2],[3,4],[9,6],[2,8]])

A*B



def mMatrixMatrix(matrix1,matrix2):
    answer = list()
    for count3 in range(len(matrix1)):
        answer.append(mVectorMatrix(matrix1[count3],matrix2))
    return answer
        
    
mMatrixMatrix(a,b)



a=[[1,2,3],[4,5,6]]
b=[[1,4],[2,5],[3,6]]


A = np.matrix([[1,2,3],[4,5,6]])
B = np.matrix([[1,4],[2,5],[3,6]])

    
    
def mIdentityMatrix(Dimension):   
    return [ [ 1 if col == row else 0 for col in range(Dimension) ] for row in range(Dimension) ]


mIdentityMatrix(3)



































        
        
