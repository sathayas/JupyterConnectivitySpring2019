import os
import numpy as np

def mutual_information(X, bins):
    A = np.zeros([X.shape[1], X.shape[1]])
    for i in range(X.shape[1]):
        print('Working on row i: %d' % i)
        for j in range(i+1, X.shape[1]):
            x1 = X[:,i]
            x2 = X[:,j]
            A[i,j] = calc_MI(x1,x2,bins)
            A[j,i] = A[i,j]
    return A


def calc_MI(X,Y,bins):

   c_XY = np.histogram2d(X,Y,bins)[0]
   c_X = np.histogram(X,bins)[0]
   c_Y = np.histogram(Y,bins)[0]

   H_X = shan_entropy(c_X)
   H_Y = shan_entropy(c_Y)
   H_XY = shan_entropy(c_XY)

   MI = H_X + H_Y - H_XY
   return MI

def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized* np.log2(c_normalized))  
    return H


