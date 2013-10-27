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
