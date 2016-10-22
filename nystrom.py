# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 21:53:08 2016

Perform Nystrom Spectral Clustering

ref: Fowlkes, Charless, et al. "Spectral grouping using the Nystrom method." 
     IEEE transactions on pattern analysis and machine intelligence 26.2 (2004): 214-225.

@author: bo

"""

import numpy as np
from sklearn import metrics
#from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
import cPickle, gzip
import sys

def nystrom(data, K):
    """
    data: input data matrix, with size n-by-d.    
    K: sample size
    
    """
    N, D = data.shape
    assert K <= N
    
    pos = np.random.choice(N, size = K, replace = False)
    idx = np.zeros(N, dtype = np.bool)
    idx[pos] = True
    
    # constructing this way, A is guaranteed to be PSD
    A = np.dot(data[idx], data[idx].T)
    B = np.dot(data[idx], data[np.logical_not(idx)].T)
    
    eigs, V = np.linalg.eig(A)
    eigs = np.real(eigs)
    # add a small constant to improve numerical stability
    # this might be problematic, since pseudo inverse is used in the paper
    A_neg_half = np.dot(V, np.diag((eigs + 1e-8)**(-0.5)))
    A_neg_half = np.dot(A_neg_half, V.T)
    
    S = A + np.dot(np.dot(A_neg_half, np.dot(B, B.T)), A_neg_half)
    eigs_S, U_S = np.linalg.eig(S)
    eigs_S = np.real(eigs_S)
    
    tmp = np.dot(np.concatenate((A, B.T), axis = 0), A_neg_half)
    V = np.dot(np.dot(tmp, U_S), np.diag((eigs_S + 1e-8)**(-0.5)))
    
    return V
    
if __name__ == '__main__':
    M = 3000;
    K = 4
    
    dataset = 'data-0.pkl.gz'
    path = '/home/bo/Data/RCV1/Processed/'
    f = gzip.open(path + dataset, 'rb')
    data = cPickle.load(f)
    f.close()
    
    train_x = data[0].toarray()
    train_x = train_x.astype(np.float32)
    train_y = np.asarray(data[1], dtype = np.int32)
    train_y = np.reshape(train_y, (train_y.shape[0], 1))
    
    dim = train_x.shape[1]
    data = np.concatenate((train_x, train_y), axis = 1)
    np.random.shuffle(data) 
    
    train_x = data[:][:, 0:dim]
    train_y = np.int32(np.squeeze(data[:][:, -1]))  
    
    V = nystrom(train_x, M)
    V = V[:][:, 0:K]
    
    km = KMeans(n_clusters = 4)
    ypred = km.fit_predict(V)
    
    nmi = metrics.normalized_mutual_info_score(train_y, ypred)
    print >> sys.stderr, ('NMI for deep clustering: %.2f' % (nmi))