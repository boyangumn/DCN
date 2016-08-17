# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 13:17:42 2016

Perform Monto-Calro simulations of: KM, SC, SNMF, DCN (deep clustering network) and NJ-DCN (non-joint, SAE + KM)

The experiment with SNMF is done by saving the data files, and run SNMF with MATLAB.

@author: yang4173
"""

import os
import numpy as np
import gzip
import cPickle
import matplotlib.pyplot as plt
from scipy.io import savemat
import sys
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import NMF
from multi_layer_km import test_SdC
from multi_layer_km_nj import test_SdC_NJ

def sigmoid(x):
    return 1/(1 + np.exp(-x))
    
      
    
    
# number of trials
N = 5

nClass = 4;
num = 1000;
sigma = 2
dim = 100

c = 10*np.array([[1,1], [0,0], [1, 0], [0, 1]])

nmi_km = np.zeros(N)
ari_km = np.zeros(N)

nmi_sc = np.zeros(N)
ari_sc = np.zeros(N)

nmi_nj = np.zeros(N)
ari_nj = np.zeros(N)

nmi_dc = np.zeros(N)
ari_dc = np.zeros(N)


data_folder = 'data'
for n in range(N):
    lowD_x  = np.zeros((nClass*num, 2))
    train_y = np.zeros((nClass*num, 1))
    for i in xrange(nClass):
        lowD_x[i*num: (i+1) *num] = np.tile(c[i,:], (num,1)) + sigma*np.random.randn(num, 2)
        # Class lables: 0, 1, 2...
        train_y[i*num: (i+1) *num] = i*np.ones((num, 1))
        
    train_y0 = train_y
    
    W = np.random.randn(100, 2)
    train_x = np.power(sigmoid(np.dot(lowD_x, W.T)), 2)
    
#    W = np.random.randn(100, 2)
#    train_x = np.tanh(sigmoid(np.dot(lowD_x, W.T)))
    
    data = np.concatenate((train_x, train_y), axis = 1)
    np.random.shuffle(data)
    train_x = data[:][:,0:dim]
    train_y = np.int32(data[:][:, -1])
    
    # save the data
    os.chdir(data_folder)
    savemat('data_'+str(n)+'.mat', {'train_x':train_x, 'train_y': train_y})
    os.chdir('../')
    
    ## Perform KMeans
    km = KMeans(n_clusters= nClass, init='k-means++', n_init=10)
    ypred = km.fit_predict(train_x)
    nmi_km[n] = metrics.adjusted_mutual_info_score(train_y, ypred)
    ari_km[n] = metrics.adjusted_rand_score(train_y, ypred)
    
    ## Perform spectral clustering
    sc = SpectralClustering(n_clusters= nClass, n_init=10, gamma=0.1, affinity='rbf', assign_labels='kmeans')
    ypred = sc.fit_predict(train_x)
    nmi_sc[n] = metrics.adjusted_mutual_info_score(train_y, ypred)
    ari_sc[n] = metrics.adjusted_rand_score(train_y, ypred)
    
    train_set = train_x, train_y
    dataset = [train_set, train_set, train_set]

    f = gzip.open('toy.pkl.gz','wb')
    cPickle.dump(dataset, f, protocol=2)
    f.close()
    ## Perform non-joint SAE+KM
    nmi_nj[n], ari_nj[n] = test_SdC_NJ(lbd = 0, finetune_lr= .01, mu = 0.9, pretraining_epochs=50,
             pretrain_lr=.01, training_epochs=100,
             dataset='toy.pkl.gz', batch_size=20, nClass = nClass, hidden_dim = [100, 50, 10, 2])    
    
    ## Perform proposed
    nmi_dc[n], ari_dc[n] = test_SdC(lbd = 0.2, finetune_lr= .01, mu = 0.9, pretraining_epochs=50,
             pretrain_lr=0.01, training_epochs=100,
             dataset='toy.pkl.gz', batch_size=20, nClass = nClass, 
             hidden_dim = [100, 50, 10, 2])             
                          
result = np.concatenate((np.mean(nmi_km, keepdims=True), np.mean(ari_km, keepdims=True), np.mean(nmi_sc, keepdims=True),
          np.mean(ari_sc, keepdims=True), np.mean(nmi_nj, keepdims=True), np.mean(ari_nj, keepdims=True),
          np.mean(nmi_dc, keepdims=True), np.mean(ari_dc, keepdims=True)) )
f = gzip.open('MC_results.pkl.gz','wb')
cPickle.dump(result, f, protocol=2)
f.close()
    

    
    
    
    
    