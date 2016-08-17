# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 12:09:48 2016

@author: bo

Create a toy dataset, including train_set



"""

import numpy as np
import gzip
import cPickle
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import NMF
from multi_layer_km import test_SdC
from multi_layer_km_nj import test_SdC_NJ
# size of the 3 sets

def sigmoid(x):
    return 1/(1 + np.exp(-x))

nClass = 4
num = 1000
sigma = 1.5
dim = 100

c = 10*np.array([[1,1], [0,0], [1, 0], [0, 1]])

np.random.seed(1)

lowD_x  = np.zeros((nClass*num, 2))
train_y = np.zeros((nClass*num, 1))
for i in xrange(nClass):
    lowD_x[i*num: (i+1) *num] = np.tile(c[i,:], (num,1)) + sigma*np.random.randn(num, 2)
    # Class lables: 0, 1, 2...
    train_y[i*num: (i+1) *num] = i*np.ones((num, 1))
    
train_y0 = train_y

# tanh(Wx), linear mapping with tanh() nonlinearity

W = np.random.randn(100, 2)
train_x = sigmoid(np.fmax(0, np.dot(lowD_x, W.T)))


#W1 = np.random.randn(10, 2)
#W2 = np.random.randn(100, 10)
#t1 = sigmoid(np.dot(lowD_x, W1.T))
#t2 = sigmoid(np.dot(t1, W2.T))
#train_x = t2

#W = np.random.randn(100, 2)
#train_x = np.tanh(sigmoid(np.dot(lowD_x, W.T)))

#W = np.random.randn(100, 2)
#train_x = np.power(sigmoid(np.dot(lowD_x, W.T)), 2)

#l1 = np.maximum(np.dot(lowD_x, W.T), 0)
#train_x = np.tanh(np.dot(l1, np.random.randn(dim, dim)))
#train_x = train_x/np.amax(train_x)

### Circle data
#theta = np.linspace(0, 2 * np.pi, num)
#lowD_x = np.zeros((nClass*num, 2))
## class 1 center: (2, 2), radius: 1
#lowD_x[0: num] = np.array([2 + np.sin(theta), 2 + np.cos(theta)]).T
#lowD_x[num: 2*num] = np.array([2 + 1.5*np.sin(theta), 2 + 1.5*np.cos(theta)]).T
#train_x = lowD_x

# shuffling 
data = np.concatenate((train_x, train_y), axis = 1)
np.random.shuffle(data)

train_x = data[:][:,0:dim] + 0.0 * np.random.randn(nClass*num, dim)
train_y = np.int32(data[:][:, -1])

## find the result of PCA
## centering
center_x = train_x - np.tile(np.mean(train_x, axis = 0), (nClass*num, 1))
# svd
U, S, V = np.linalg.svd(center_x)
#
## Calculate rank-2 reconstruction error
#A = np.dot(U[:][:, 0:2], np.diag(S[0:2]))
#AA = np.dot(A, V[0:2])
#Err = np.mean(np.sum(np.power(center_x - AA, 2), axis = 1))
#
#print >> sys.stderr, ('Average squared rank-2 PCA reconstruction error: %.4f' %Err) 
#
## Perform a NMF
nmf_model = NMF(n_components=2, init='random')
WW = nmf_model.fit_transform(train_x)

color = ['b', 'g', 'r', 'm', 'k', 'b', 'g', 'r', 'm', 'k']
marker = ['o', '+','o', '+','o', '+','o', '+','o', '+']
# show ground-truth
data_to_plot = lowD_x
for i in xrange(nClass):
    idx_x = data_to_plot[np.nonzero(train_y0 == i), 0]
    idx_y = data_to_plot[np.nonzero(train_y0 == i), 1]  
    plt.figure(0)      
    plt.scatter(idx_x, idx_y, s = 70, c = color[i], marker = marker[i], label = '%s'%i)
plt.legend()    
plt.show()

## show result by PCA
data_to_plot = U
for i in xrange(nClass):
    idx_x = data_to_plot[np.nonzero(train_y == i), 0]
    idx_y = data_to_plot[np.nonzero(train_y == i), 1]   
    plt.figure(1)     
    plt.scatter(idx_x, idx_y, s = 70, c = color[i], marker = marker[i], label = '%s'%i)
plt.legend()    
plt.show()
#
## show result by NMF
data_to_plot = WW
for i in xrange(nClass):
    idx_x = data_to_plot[np.nonzero(train_y == i), 0]
    idx_y = data_to_plot[np.nonzero(train_y == i), 1]   
    plt.figure(2)     
    plt.scatter(idx_x, idx_y, s = 70, c = color[i], marker = marker[i], label = '%s'%i)
plt.legend()    
plt.show()


## Perform spectral clustering
sc = SpectralClustering(n_clusters= nClass, n_init=10, gamma=0.1, affinity='rbf', 
                        n_neighbors=3, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None)
ypred = sc.fit_predict(train_x)
nmi_sc = metrics.adjusted_mutual_info_score(train_y, ypred)
ari_sc = metrics.adjusted_rand_score(train_y, ypred)
print >> sys.stderr, ('NMI for spectral clustering: %.2f' % (nmi_sc))
print >> sys.stderr, ('ARI for spectral clustering: %.2f' % (ari_sc)) 


## Perform KMeans
km = KMeans(n_clusters= nClass, init='k-means++', n_init=10)
ypred = km.fit_predict(train_x)
nmi_km = metrics.adjusted_mutual_info_score(train_y, ypred)
ari_km = metrics.adjusted_rand_score(train_y, ypred)
print >> sys.stderr, ('NMI for Kmeans: %.2f' % (nmi_km))
print >> sys.stderr, ('ARI for Kmeans: %.2f' % (ari_km))


train_set = train_x, train_y
dataset = [train_set, train_set, train_set]

f = gzip.open('toy.pkl.gz','wb')
cPickle.dump(dataset, f, protocol=2)
f.close()

nmi_dc, ari_dc = test_SdC(lbd = .1, finetune_lr= .05, mu = 0.9, pretraining_epochs=50,
             pretrain_lr=0.01, training_epochs=100,
             dataset='toy.pkl.gz', batch_size=20, nClass = nClass, 
             hidden_dim = [100, 50, 10, 2]) 
#             
print >> sys.stderr, ('NMI for spectral clustering: %.2f' % (nmi_sc))
print >> sys.stderr, ('ARI for spectral clustering: %.2f' % (ari_sc))

print >> sys.stderr, ('NMI for deep clustering: %.2f' % (nmi_dc))
print >> sys.stderr, ('ARI for deep clustering: %.2f' % (ari_dc))



#nmi_nj, ari_nj = test_SdC_NJ(lbd = 0, finetune_lr= .01, mu = 0.9, pretraining_epochs=50,
#             pretrain_lr=.01, training_epochs=100,
#             dataset='toy.pkl.gz', batch_size=20, nClass = nClass, hidden_dim = [100, 50, 10, 2])
#
#print >> sys.stderr, ('NMI for SAE + Kmeans: %.2f' % (nmi_nj))
#print >> sys.stderr, ('ARI for SAE + Kmeans: %.2f' % (ari_nj))

## Working configuration
## W = np.random.randn(100, 2)
## train_x = np.power(sigmoid(np.dot(lowD_x, W.T)), 2)
#
#test_SdC(lbd = .1, finetune_lr= .01, mu = 0.9, pretraining_epochs=50,
#             pretrain_lr=0.01, training_epochs=100,
#             dataset='toy.pkl.gz', batch_size=20, nClass = nClass, 
#             hidden_dim = [100, 50, 10, 2]) 
##             
#print >> sys.stderr, ('NMI for spectral clustering: %.2f' % (nmi_sc))
#print >> sys.stderr, ('ARI for spectral clustering: %.2f' % (ari_sc))



## Working configuration
## W = np.random.randn(100, 2)
## train_x = np.tanh(sigmoid(np.dot(lowD_x, W.T)))

#test_SdC(lbd = 0.2, finetune_lr= .01, mu = 0.9, pretraining_epochs=50,
#             pretrain_lr=0.5, training_epochs=100,
#             dataset='toy.pkl.gz', batch_size=20, nClass = nClass, 
#             hidden_dim = [100, 50, 10, 2]) 
#print >> sys.stderr, ('NMI for spectral clustering: %.2f' % (nmi_sc))
#print >> sys.stderr, ('ARI for spectral clustering: %.2f' % (ari_sc))        


## Working configuration
     
#W1 = np.random.randn(10, 2)
#W2 = np.random.randn(100, 10)
#t1 = sigmoid(np.dot(lowD_x, W1.T))
#t2 = sigmoid(np.dot(t1, W2.T))
#train_x = t2
# nmi_dc, ari_dc = test_SdC(lbd = 1, finetune_lr= .05, mu = 0.9, pretraining_epochs=50,
#         pretrain_lr=0.01, training_epochs=100,
#         dataset='toy.pkl.gz', batch_size=20, nClass = nClass, 
#         hidden_dim = [100, 50, 10, 2]) 
