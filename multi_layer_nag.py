# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 14:27:50 2016

@author: bo

Multiple-layers Deep Clustering

06/19/2016 Multi-layer autoencoder, without reconstruction, performance is not good, as expected.

06/20/2016 Multi-layer autoencoder, with reconstruction and clustering as loss, seems to give meaningful result on MNIST

06/21/2016 Modified cost output, so that the functions print out cost for both reconstruction and clustering, 
            added an input lbd, to enable tuning parameter that balancing the two costs--not an easy job.

06/29/2016 Changed how learning-rate (stepsize, both pretraining and finetuning) and center_array are passed and manipulated
           by using shared-variable mechanism in Theano. Now the stepsize is diminishing c/sqrt(t), where c is some fixed constant            
"""

import os
import sys
import timeit

import numpy 
import cPickle
import gzip
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
 
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
#from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
from utils import tile_raster_images

#from logistic_sgd import LogisticRegression
#from mlp import HiddenLayer
from dA import dA
from deepclustering import load_data
from mlp import HiddenLayer
from multi_layer import SdC, dA2

try:
    import PIL.Image as Image
except ImportError:
    import Image
    
        
def test_SdC(lbd = .01, finetune_lr= .005, mu = 0.9, pretraining_epochs=50,
             pretrain_lr=.001, training_epochs=150,
             dataset='toy.pkl.gz', batch_size=20, nClass = 4, hidden_dim = [100, 50, 2]):
    """
    Demonstrates how to train and test a stochastic denoising autoencoder.

    This is demonstrated on MNIST.
    
    :type lbd: float
    :param lbd: tuning parameter, multiplied on reconstruction error, i.e. the larger
                lbd the larger weight on minimizing reconstruction error.
                
    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    (factor for the stochastic gradient)

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type n_iter: int
    :param n_iter: maximal number of iterations ot run the optimizer

    :type dataset: string
    :param dataset: path the the pickled dataset

    """

    datasets = load_data(dataset)  

    train_set_x, train_set_y = datasets[0]
#    valid_set_x, valid_set_y = datasets[1]
#    test_set_x,  test_set_y  = datasets[2]
    
    inDim = train_set_x.get_value().shape[1]
    label_true = numpy.int32(train_set_y.get_value(borrow=True))
    
    index = T.lscalar() 
    x = T.matrix('x')
    
#    x.tag.test_value = numpy.random.rand(50000, 784).astype('float32')
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    # numpy random generator
    # start-snippet-3
    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'
    # construct the stacked denoising autoencoder class
    sdc = SdC(
        numpy_rng=numpy_rng,
        n_ins=inDim,
        lbd = lbd, 
        input=x,
        hidden_layers_sizes= hidden_dim,
    )
    # end-snippet-3 start-snippet-4
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = sdc.pretraining_functions(train_set_x=train_set_x,
                                                batch_size=batch_size)

    print '... pre-training the model'
    start_time = timeit.default_timer()
    ## Pre-train layer-wise
    corruption_levels = [.0, .0, .0, 0, 0]
    
    pretrain_lr_shared = theano.shared(numpy.asarray(pretrain_lr,
                                                   dtype='float32'),
                                     borrow=True)
    for i in xrange(sdc.n_layers):
        # go through pretraining epochs
        iter = 0
        for epoch in xrange(pretraining_epochs):
            # go through the training set  
            c = []  
            for batch_index in xrange(n_train_batches):
                iter = (epoch) * n_train_batches + batch_index 
                pretrain_lr_shared.set_value( numpy.float32(pretrain_lr) )
#                pretrain_lr_shared.set_value( numpy.float32(pretrain_lr/numpy.sqrt(iter + 1)) )
                cost = pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr_shared.get_value())                         
                c.append(cost)
                
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)

    end_time = timeit.default_timer()

    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    # end-snippet-4
    ########################
    # FINETUNING THE MODEL #
    ########################

    
    km = MiniBatchKMeans(n_clusters = nClass, batch_size=100)   
    
    out = sdc.get_output()
    out_sdc = theano.function(
        [index],
        outputs = out,
        givens = {x: train_set_x[index * batch_size: (index + 1) * batch_size]}
    )    
    hidden_val = [] 
    for batch_index in xrange(n_train_batches):
         hidden_val.append(out_sdc(batch_index))
    
    hidden_array  = numpy.asarray(hidden_val)
    hidden_size = hidden_array.shape        
    hidden_array = numpy.reshape(hidden_array, (hidden_size[0] * hidden_size[1], hidden_size[2] ))
      
    # use the true labels to get initial cluster centers
    centers = numpy.zeros((nClass, hidden_size[2]))
    
    for i in xrange(nClass):
        temp = hidden_array[label_true == i]        
        centers[i] = numpy.mean(temp, axis = 0)      
    
    center_array = centers[label_true]
#    # Do a k-means clusering to get center_array  
#    ypred = km.fit_predict(hidden_array)
#    center_array = km.cluster_centers_[[km.labels_]]             
    center_shared =  theano.shared(numpy.asarray(center_array ,
                                                   dtype='float32'),
                                     borrow=True)
    lr_shared = theano.shared(numpy.asarray(finetune_lr,
                                                   dtype='float32'),
                                     borrow=True)

    print '... getting the finetuning functions'   
       
    train_fn = sdc.build_finetune_functions(
        datasets=datasets,
        center_shared=center_shared,
        batch_size=batch_size,
        mu = mu,
        learning_rate=lr_shared
    )

    print '... finetunning the model'
    # early-stopping parameters

    start_time = timeit.default_timer()
    done_looping = False
    epoch = 0

    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1    
        c = [] # total cost
        d = [] # cost of reconstruction    
        e = [] # cost of clustering 
        f = [] # learning_rate
        g = []
        for minibatch_index in xrange(n_train_batches):
            # calculate the stepsize
            iter = (epoch - 1) * n_train_batches + minibatch_index 
            lr_shared.set_value( numpy.float32(finetune_lr) )
#            lr_shared.set_value( numpy.float32(finetune_lr/numpy.sqrt(epoch)) )
            cost = train_fn(minibatch_index)
#            aa = sdc.sigmoid_layers[0].W.get_value()
            c.append(cost[0])
            d.append(cost[1])
            e.append(cost[2])
            f.append(cost[3])
#            gg = cost[4]
#            g.append(gg)
            
        # Do a k-means clusering to get center_array    
        hidden_val = [] 
        for batch_index in xrange(n_train_batches):
             hidden_val.append(out_sdc(batch_index))
        
        hidden_array  = numpy.asarray(hidden_val)
        hidden_size = hidden_array.shape        
        hidden_array = numpy.reshape(hidden_array, (hidden_size[0] * hidden_size[1], hidden_size[2] ))
        km.fit(hidden_array)
        center_array = km.cluster_centers_[[km.labels_]]   
        center_shared.set_value(numpy.asarray(center_array, dtype='float32'))          
#        center_shared =  theano.shared(numpy.asarray(center_array ,
#                                                       dtype='float32'),
#                                         borrow=True)   
        print 'Fine-tuning epoch %d ++++ \n' % (epoch), 
        print ('Total cost: %.5f, '%(numpy.mean(c)) + 'Reconstruction: %.5f, ' %(numpy.mean(d)) 
            + "Clustering: %.5f, " %(numpy.mean(e)) )
#        print 'Learning rate: %.6f' %numpy.mean(f)

    err = numpy.mean(d)
    print >> sys.stderr, ('Average squared 2-D reconstruction error: %.4f' %err)
    end_time = timeit.default_timer()
    ypred = km.predict(hidden_array)   
    nmi_dc = metrics.adjusted_mutual_info_score(label_true, ypred)
    print >> sys.stderr, ('NMI for deep clustering: %.2f' % (nmi_dc))

    ari_dc = metrics.adjusted_rand_score(label_true, ypred)
    print >> sys.stderr, ('ARI for deep clustering: %.2f' % (nmi_dc))
#    print(
#        (
#            'Optimization complete with best validation score of %f %%, '
#            'on iteration %i, '
#            'with test performance %f %%'
#        )
#        % (best_validation_loss * 100., best_iter + 1, test_score * 100.)
#    )
    
    f = open('deepclus.save', 'wb')
    cPickle.dump([param.get_value() for param in sdc.params], f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    color = ['b', 'g', 'r', 'm', 'k', 'b', 'g', 'r', 'm', 'k']
    marker = ['o', '+','o', '+','o', '+','o', '+','o', '+']
    
    # Take 500 samples to plot
    data_to_plot = hidden_array[0:1999]
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

  
    if dataset == 'toy.pkl.gz':
        x = train_set_x.get_value()[:, 0]
        y = train_set_x.get_value()[:, 1]
        
        # using resulted label, and the original data
        pred_label = ypred[0:1999]
        for i in xrange(nClass):
            idx_x = x[numpy.nonzero( pred_label == i)]
            idx_y = y[numpy.nonzero( pred_label == i)]   
            plt.figure(4)
            plt.scatter(idx_x, idx_y, s = 70, c = color[i], marker = marker[i], label = '%s'%i)
        
        plt.legend()
        plt.show() 
           
    

if __name__ == '__main__':      
    test_SdC(lbd = 1, finetune_lr= .1, mu = 0.9, pretraining_epochs=50,
             pretrain_lr=1, training_epochs=100,
             dataset='toy.pkl.gz', batch_size=20, nClass = 4, 
             hidden_dim = [100, 50, 20])      
    
        