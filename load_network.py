# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 09:29:43 2016

@author: yang4173

This script loads a saved network, calculate the learned representation and save for future use.


"""
import cPickle
# repeated definition of SdC, take action!
from multi_layer_rbm import SdC
from deepclustering import load_data
import numpy
import matplotlib.pyplot as plt

saved_network = 'deepclus.save'
dataset='mnist.pkl.gz'
nClass = 10

f = open(saved_network, 'rb')
param_init = cPickle.load(f)
f.close()

datasets = load_data(dataset)  
train_set_x, train_set_y = datasets[0]
data = train_set_x.get_value() 
label_true = train_set_y.get_value() 

inDim = data.shape[1]     
# find a better way to save and load model params
lbd = 1
hidden_dim = [1000, 500, 250, 2]

numpy_rng = numpy.random.RandomState(125)    
sdc = SdC(
        numpy_rng=numpy_rng,
        n_ins=inDim,
        lbd = lbd, 
        input=data,
        hidden_layers_sizes= hidden_dim,
        Param_init = param_init
    )
Output = sdc.get_output().eval()

f = open('LearnedRep.save', 'wb')
cPickle.dump(Output, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

print 'Done'

color = ['b', 'g', 'r', 'm', 'k', 'b', 'g', 'r', 'm', 'k']
marker = ['o', '+','o', '+','o', '+','o', '+','o', '+']

# Take 500 samples to plot
data_to_plot = Output[0:1999]
label_plot = label_true[0:1999]   

x = data_to_plot[:, 0]
y = data_to_plot[:, 1]

for i in xrange(nClass):
    idx_x = x[numpy.nonzero(label_plot == i)]
    idx_y = y[numpy.nonzero(label_plot == i)]   
    plt.figure(3)
    plt.scatter(idx_x, idx_y, s = 70, c = color[i], marker = marker[i], label = '%s'%i)

plt.legend()
plt.show() 




