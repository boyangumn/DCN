# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 09:29:43 2016

@author: yang4173

This script loads a saved network, calculate the learned representation and save for future use.


"""
import cPickle, gzip
import os, sys
from multi_layer_km import SdC, load_rcv
from deepclustering import load_data
import numpy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from mnist_loader import MNIST
from cluster_acc import acc
from sklearn import svm
import theano.tensor as T
import theano

#saved_network = 'deepclus.save'
#dataset='mnist.pkl.gz'

os.chdir('./MNIST_results/Finalized/')
#saved_network = 'deepclus_'+str(nClass)+ '_clusters.pkl.gz'
saved_network = 'deepclus_10_clusters.pkl.gz'
with gzip.open(saved_network, 'rb') as f:
    saved_result = cPickle.load(f)

param_init = saved_result['network']
hidden_dim = [2000, 1000, 1000, 1000, 50]
lbd = 1
#hidden_dim = saved_result['config']['hidden_dim']
#lbd = saved_result['config']['lbd']


## MNIST data
#datapath = '/home/bo/Data/MNIST/'
#filename = 'mnist.pkl.gz'
#with gzip.open(datapath + filename, 'rb') as f:
#    train_set, test_set, valid_set = cPickle.load(f)
#train_x = numpy.concatenate((train_set[0], test_set[0], valid_set[0]), axis = 0)
#train_y = numpy.concatenate((train_set[1], test_set[1], valid_set[1]), axis = 0)
#inDim = train_x.shape[1]


## infimnist data
#datapath = '/home/bo/Data/infimnist/'
#path_img = datapath + 'mnist500k-images-idx1-ubyte'
#path_lbl = datapath + 'mnist500k-labels-idx1-ubyte'
#train_x, train_y = MNIST.load(path_img, path_lbl)
#
#data = train_x
#label_true = train_y
#inDim = data.shape[1]
#datasets = load_data(dataset)  
#train_set_x, train_set_y = datasets[0]
#data = train_set_x.get_value() 
#label_true = train_set_y.get_value() 
#
#inDim = data.shape[1]     
## find a better way to save and load model params
#lbd = 1
#hidden_dim = [1000, 500, 250, 2]

## RCV1

datapath = '/home/bo/Data/RCV1/Processed/'
filename = 'data-0.pkl.gz'
batch_size = 100
datasets = load_rcv(datapath + filename, batch_size)     
train_set_x,  train_set_y  = datasets[0]    
train_y = numpy.squeeze(train_set_y.get_value())
inDim = train_set_x.get_value().shape[1]

numpy_rng = numpy.random.RandomState(125)    
x = T.matrix('x')
index = T.lscalar() 

sdc = SdC(
        numpy_rng=numpy_rng,
        n_ins=inDim,
        lbd = lbd, 
        input = x,
        hidden_layers_sizes= hidden_dim,
        Param_init = param_init
    )
out = sdc.get_output()
out_sdc = theano.function(
        [index],
        outputs = out,
        givens = {x: train_set_x[index * batch_size: (index + 1) * batch_size]}
    ) 
hidden_val = [] 
N = train_set_x.get_value(borrow=True).shape[0]
n_train_batches = N/batch_size

for batch_index in xrange(n_train_batches):
     hidden_val.append(out_sdc(batch_index))
hidden_array  = numpy.asarray(hidden_val)
hidden_size = hidden_array.shape        
hidden_array = numpy.reshape(hidden_array, (hidden_size[0] * hidden_size[1], hidden_size[2] ))

### Train a SVM classifier
train_pct = 0.8
train_num = numpy.floor(N*0.9)

svm_train_x = hidden_array[0:train_num]
svm_test_x = hidden_array[train_num:]

svm_train_y = train_y[0:train_num]
svm_test_y = train_y[train_num:]

svm_model = svm.SVC(kernel = 'linear')
svm_model.fit(svm_train_x, svm_train_y)

ypred = svm_model.predict(svm_test_x)
ac = 1.0*numpy.count_nonzero(numpy.equal(ypred, svm_test_y))/svm_test_y.shape[0]
print >> sys.stderr, ('Acc for classification is: %.2f' % (ac))

### Do a Kmeans clustering
#km = KMeans(n_clusters = nClass)  
#ypred = km.fit_predict(Output)
#
#nmi = metrics.normalized_mutual_info_score(train_y, ypred)
#print >> sys.stderr, ('NMI for deep clustering: %.2f' % (nmi))
#
#ari = metrics.adjusted_rand_score(train_y, ypred)
#print >> sys.stderr, ('ARI for deep clustering: %.2f' % (ari))
#
#try:
#    ac = acc(ypred, train_y)
#except AssertionError:
#    ac = 0
#    print('Number of predicted cluster mismatch with ground truth.')
#    
#print >> sys.stderr, ('Acc for deep clustering: %.2f' % (ac))


#f = open('LearnedRep.save', 'wb')
#cPickle.dump(Output, f, protocol=cPickle.HIGHEST_PROTOCOL)
#f.close()
#
#print 'Done'
#
#color = ['b', 'g', 'r', 'm', 'k', 'b', 'g', 'r', 'm', 'k']
#marker = ['o', '+','o', '+','o', '+','o', '+','o', '+']
#
## Take 500 samples to plot
#data_to_plot = Output[0:1999]
#label_plot = label_true[0:1999]   
#
#x = data_to_plot[:, 0]
#y = data_to_plot[:, 1]
#
#for i in xrange(nClass):
#    idx_x = x[numpy.nonzero(label_plot == i)]
#    idx_y = y[numpy.nonzero(label_plot == i)]   
#    plt.figure(3)
#    plt.scatter(idx_x, idx_y, s = 70, c = color[i], marker = marker[i], label = '%s'%i)
#
#plt.legend()
#plt.show() 




