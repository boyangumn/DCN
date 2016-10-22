# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 14:03:37 2016

Try out SC on MNIST

@author: yang4173
"""

from sklearn.cluster import SpectralClustering
import scipy.io as sio
from sklearn import metrics
from sklearn.neighbors import kneighbors_graph
from sklearn.manifold import spectral_embedding
from sklearn.cluster import KMeans
import gzip 
import cPickle
import numpy


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

# perform SC on the test set
data_x, data_y = test_set

k = 12
nClass = 500
A = kneighbors_graph(data_x, k)
V = spectral_embedding(A, n_components = 10, drop_first = False)
V = V + numpy.absolute(numpy.min(V))
#V = V/numpy.amax(V)
#
km_model = KMeans(n_clusters = nClass)
ypred = km_model.fit_predict(V)
nmi = metrics.normalized_mutual_info_score(data_y, ypred)
print('The NMI is: %.4f'%nmi)
#
V = numpy.float32(V)

f = gzip.open('EVD-test500.pkl.gz', 'wb')
cPickle.dump([(V, data_y), 0, 0], f, protocol = 2)
f.close()
#sio.savemat('V_train_10.mat', {'train_x': V, 'train_y': data_y})


#sc = SpectralClustering(n_clusters = nClass, affinity = 'nearest_neighbors', n_neighbors = k)
#sc = SpectralClustering(n_clusters = 10, affinity = 'rbf',gamma = 1, n_neighbors = 10)

#data_x = data_x[0:1000]
#data_y = data_y[0:1000]

#ypred = sc.fit_predict(data_x)
#nmi = metrics.normalized_mutual_info_score(data_y, ypred)
#ari = metrics.adjusted_rand_score(data_y, ypred)
#print 'NMI is: %.4f' %nmi
#print 'ARI is: %.4f' %ari