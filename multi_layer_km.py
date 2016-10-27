# -*- coding: utf-8 -*-
"""
@author: bo

Multiple-layers Deep Clustering
         
"""

import os
import sys
import timeit
import scipy
import numpy 
import cPickle
import gzip
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from theano.tensor.shared_randomstreams import RandomStreams
from cluster_acc import acc 
from sklearn import metrics
from sklearn.cluster import KMeans
from dA import dA

class dA2(dA):
    # overload the original function in dA class
    # using the ReLU nonlinearity
    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        bhid=None,
        bvis=None,
        gamma = None,
        beta = None
    ):
        """
        Initialize the dA2 class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the corruption level. The
        constructor also receives symbolic variables for the input, weights and
        bias. 

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone dA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None
                    
        :type gamma: theano.tensor.TensorType
        :param gamma: Tensor variable for implementing batch normalization
        
        :type beta: theano.tensor.TensorType
        :param beta: Tensor variable for implementing batch normalization

        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if W is None:
            
            initial_W = numpy.asarray(
                0.01*numpy.float32(numpy.random.randn(n_visible, n_hidden))               
                
            )
        else:
            initial_W = W            
        W = theano.shared(value=initial_W, name='W', borrow=True)

        if bvis is None:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )
        else:
            bvis = theano.shared(
                value=bvis,
                borrow=True
            )

        if bhid is None:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )
        else:
            bhid = theano.shared(
                value=bhid,
                name='b',
                borrow=True
            )
            
        if gamma is None:
            gamma = theano.shared(value = numpy.ones((n_hidden,), dtype=theano.config.floatX), name='gamma')
        else:
            gamma = theano.shared(value = gamma, name='gamma')
                                    
        if beta is None:
            beta = theano.shared(value = numpy.zeros((n_hidden,),dtype=theano.config.floatX), name='beta')
        else:
            beta = theano.shared(value = beta, name='beta')

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        self.gamma = gamma
        self.beta = beta
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input
                
        self.params = [self.W, self.b, self.b_prime, self.gamma, self.beta]
        # delta is a temporary variable for implementing the momentum method
        self.delta = [theano.shared(value = numpy.zeros((n_visible, n_hidden), dtype = theano.config.floatX), borrow=True), 
                       theano.shared(value = numpy.zeros(n_hidden,  dtype = theano.config.floatX), borrow = True ),
                        theano.shared(value = numpy.zeros(n_visible,  dtype = theano.config.floatX), borrow = True ),
                        theano.shared(value = numpy.zeros(n_hidden,  dtype = theano.config.floatX), borrow = True ),
                        theano.shared(value = numpy.zeros(n_hidden,  dtype = theano.config.floatX), borrow = True )
                     ]        
        
    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        
        linear = T.dot(input, self.W) + self.b
        bn_output = T.nnet.bn.batch_normalization(inputs = linear,
			gamma = self.gamma, beta = self.beta, mean = linear.mean((0,), keepdims=True),
			std = T.ones_like(linear.var((0,), keepdims = True)), mode='high_mem')
   
        return T.nnet.relu(bn_output)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer

        """
        return T.nnet.relu(T.dot(hidden, self.W_prime) + self.b_prime)
    def get_cost_updates(self, corruption_level, learning_rate, mu):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        
        L = T.sum(T.pow(self.x - z, 2), axis = 1)                
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = []
        for param, delta, gparam in zip(self.params, self.delta, gparams):
            updates.append( (delta, mu*delta - learning_rate * gparam) )
            updates.append( (param, param + mu*mu*delta - (1+mu)*learning_rate*gparam ))

        return (cost, updates)                
   
class SdC(object):
    """
    
    class SdC, main class for deep-clustering network, constructed by stacking multiple dA2 layers.
    
    It is possilbe to initialize the network with a saved network trained before, just pass the network parameters 
    to Param_init. This facilites parameter tuning for the optimization part, by avoiding performing pre-training
    every time.
     
    """
    
    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input = None,
        n_ins=784,
        lbd = 1,
        beta = 1,
        hidden_layers_sizes=[1000, 200, 10],
        corruption_levels=[0, 0, 0],
        Param_init = None
    ):
#        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        self.lbd = lbd
        self.beta = beta
        self.delta = []   
    
        assert self.n_layers > 0
    
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        if input is None:
            self.x = T.matrix('x')  # the data is presented as rasterized images
        else:
            self.x = input
            
        self.y = T.ivector('y')  # the labels are presented as 1D vector of
        
        for i in xrange(self.n_layers):
            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]
    
            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.dA_layers[-1].get_hidden_values(self.dA_layers[-1].x)
            if Param_init is None:
                dA_layer = dA2(numpy_rng=numpy_rng,
                              theano_rng=theano_rng,
                              input=layer_input,
                              n_visible=input_size,
                              n_hidden=hidden_layers_sizes[i])                
            else:
                dA_layer = dA2(numpy_rng=numpy_rng,
                              theano_rng=theano_rng,
                              input=layer_input,
                              n_visible=input_size,
                              n_hidden=hidden_layers_sizes[i],
                              W = Param_init[5*i],
                              bhid = Param_init[5*i + 1],
                              bvis = Param_init[5*i+2],
                              gamma = Param_init[5*i+3],
                              beta = Param_init[5*i+4])                
                          
            self.dA_layers.append(dA_layer)
            self.params.extend(dA_layer.params)                       
            self.delta.extend(dA_layer.delta) 
                      
    def get_output(self):        
#        return self.sigmoid_layers[-1].output
        return self.dA_layers[-1].get_hidden_values(self.dA_layers[-1].x)
        
    def get_network_reconst(self):
        reconst = self.get_output()
        for da in reversed(self.dA_layers):
            reconst = T.nnet.relu(T.dot(reconst, da.W_prime) + da.b_prime)
            
        return reconst
        
    def finetune_cost_updates(self, center, mu, learning_rate):
        """ This function computes the cost and the updates ."""

        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, withd one entry per
        #        example in minibatch
        network_output = self.get_output()
        temp = T.pow(center - network_output, 2)    
        
        L =  T.sum(temp, axis=1) 
        # Add the network reconstruction error 
        z = self.get_network_reconst()
        reconst_err = T.sum(T.pow(self.x - z, 2), axis = 1)            
        L = self.beta*L + self.lbd*reconst_err
        
        cost1 = T.mean(L)
        cost2 = self.lbd*T.mean(reconst_err)  
        cost3 = cost1 - cost2

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost1, self.params)  
        # generate the list of updates
        updates = []
        grad_values = []
        param_norm = []
        for param, delta, gparam in zip(self.params, self.delta, gparams):
            updates.append( (delta, mu*delta - learning_rate * gparam) )
            updates.append( (param, param + mu*mu*delta - (1+mu)*learning_rate*gparam ))
            grad_values.append(gparam.norm(L=2))
            param_norm.append(param.norm(L=2))
        
        grad_ = T.stack(*grad_values)
        param_ = T.stack(*param_norm)
        return ((cost1, cost2, cost3, grad_, param_), updates)
        
        
    def pretraining_functions(self, train_set_x, batch_size, mu):
        ''' Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch
        
        :type mu: float
        :param mu: extrapolation parameter used for implementing Nesterov-type acceleration

        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size
        
        pretrain_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate, mu)
            # compile the theano function
            fn = theano.function(
                inputs=[
                    index,
                    theano.In(corruption_level),
                    theano.In(learning_rate)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                },
                on_unused_input='ignore'
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns    
        
    def build_finetune_functions(self, datasets, centers, batch_size, mu, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                         the has to contain three pairs, `train`,
                         `valid`, `test` in this order, where each pair
                         is formed of two Theano variables, one for the
                         datapoints, the other for the labels
                         
        :type centers: numpy ndarray
        :param centers: the centroids corresponding to each data sample in the minibatch        

        :type batch_size: int
        :param batch_size: size of a minibatch
        
        :type mu: float
        :param mu: extrapolation parameter used for implementing Nesterov-type acceleration

        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        
        '''

        (train_set_x, train_set_y) = datasets[0]
        
        index = T.lscalar('index')  # index to a [mini]batch
        minibatch = T.fmatrix('minibatch')

        # compute the gradients with respect to the model parameters
        cost, updates = self.finetune_cost_updates(
        centers, 
        mu,
        learning_rate=learning_rate
        )
        minibatch = train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ]

        train_fn = theano.function(
            inputs=[index],
            outputs= cost,
            updates=updates,
            givens={
                self.x: minibatch
            },
            name='train'
        )
        return train_fn  

def load_data(dataset):
    """
    
    Load the dataset, perform shuffling
    
    """
    with gzip.open(dataset, 'rb') as f:
        train_x, train_y = cPickle.load(f)
    if scipy.sparse.issparse(train_x):
        train_x = train_x.toarray()
    if train_x.dtype != 'float32':
        train_x = train_x.astype(numpy.float32)
    if train_y.dtype != 'int32':
        train_y = train_y.astype(numpy.int32)
    
    if train_y.ndim > 1:
        train_y = numpy.squeeze(train_y)  
    N = train_x.shape[0]
    idx = numpy.random.permutation(N)
    train_x = train_x[idx]
    train_y = train_y[idx]
    
    return train_x, train_y    
        
def load_data_shared(dataset, batch_size):
    """

    Load the dataset and save it as shared-variable to be used by Theano
    
    """
    with gzip.open(dataset, 'rb') as f:
        train_x, train_y = cPickle.load(f)
    N = train_x.shape[0] - train_x.shape[0] % batch_size
    train_x = train_x[0: N]    
    train_y = train_y[0: N]
    
    # shuffling
    idx = numpy.random.permutation(N)
    train_x = train_x[idx]    
    train_y = train_y[idx]
    
    # change sparse matrix into full, to be compatible with CUDA and Theano
    if scipy.sparse.issparse(train_x):
        train_x = train_x.toarray()
    if train_x.dtype != 'float32':
        train_x = train_x.astype(numpy.float32)
    if train_y.dtype != 'int32':
        train_y = train_y.astype(numpy.int32)
    if train_y.ndim > 1:
        train_y = numpy.squeeze(train_y)        
    
    data_x, data_y = shared_dataset((train_x, train_y))    
    rval = [(data_x, data_y), 0, 0]
    return rval    
    
def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(numpy.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(numpy.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        #return shared_x, T.cast(shared_y, 'int32')
        return shared_x, shared_y        
def batch_km(data, center, count):
    """
    Function to perform a KMeans update on a batch of data, center is the centroid 
    from last iteration.

    """
    N = data.shape[0]
    K = center.shape[0]   
    
    # update assignment    
    idx = numpy.zeros(N, dtype = numpy.int)
    for i in range(N):
        dist = numpy.inf
        ind = 0
        for j in range(K):
            temp_dist = numpy.linalg.norm(data[i] - center[j])              
            if temp_dist < dist:
                dist = temp_dist
                ind = j
        idx[i] = ind
        
    # update centriod
    center_new = center
    for i in range(N):
        c = idx[i]
        count[c] += 1
        eta = 1/count[c]
        center_new[c] = (1 - eta) * center_new[c] + eta * data[i]
        
    return idx, center_new, count
  
def test_SdC(Init = '', lbd = .01, output_dir='MNIST_results', save_file = '', beta = 1, finetune_lr= .005, mu = 0.9, pretraining_epochs=50,
             pretrain_lr=.001, training_epochs=150,
             dataset='toy.pkl.gz', batch_size=20, nClass = 4, hidden_dim = [100, 50, 2],  diminishing = True):
    """
    :type Init: string
    :param Init: a string contains the filename of a saved network, the saved network can be loaded to initialize 
                 the network. Leave this parameter be an empty string if no saved network available. If failed to 
                 find the specified file, the program will initialized the network randomly.
                 
    :type lbd: float
    :param lbd: tuning parameter, multiplied on reconstruction error, i.e. the larger
                lbd the larger weight on minimizing reconstruction error.
                
    :type output_dir: string
    :param output_dir: the location to save trained network
    
    :type save_file: string
    :param save_file: the filename to save trained network
    
    :type beta: float
    :param beta: the parameter for the clustering term, set to 0 if a pure SAE (without clustering regularization)
                 is intended.          
                
    :type finetune_lr: float
    :param finetune_lr: learning rate used in the finetune stage
    (factor for the stochastic gradient)
    
    :type mu: float
    :param mu: extrapolation parameter used for implementing Nesterov-type acceleration
    
    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training
    
    :type training_epochs: int
    :param training_epochs: number of epoch to do optimization

    :type dataset: string
    :param dataset: path of the pickled dataset
    
    :type batch_size: int
    :param batch_size: number of data samples in one minibatch
    
    :type nClass: int
    :param nClass: number of clusters
    
    :hidden dim: array
    :param hidden_dim: the number of neurons in each hidden layer in the forward network, the reconstruction part 
                       has a mirror-image structure

    :type diminishing: boolean
    :param diminishing: whether or not to reduce learning rate during optimization, if True, the learning rate is 
                        halfed every 5 epochs.
    """    
    datasets = load_data_shared(dataset, batch_size)  
    
    working_dir = os.getcwd()
    train_set_x,  train_set_y  = datasets[0]    
    inDim = train_set_x.get_value().shape[1]
    label_true = numpy.squeeze(numpy.int32(train_set_y.get_value(borrow=True)))
    
    index = T.lscalar() 
    x = T.matrix('x')
    
    # compute number of minibatches for training, validation and testing
    n_train_samples = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches = n_train_samples
    n_train_batches /= batch_size

    # numpy random generator
    # start-snippet-3
    numpy_rng = numpy.random.RandomState(89677)
#    numpy_rng = numpy.random.RandomState()
    print '... building the model'
    try:
        os.chdir(output_dir)
    except OSError:
        os.mkdir(output_dir)
        os.chdir(output_dir)
    # construct the stacked denoising autoencoder class
    if Init == '':
        sdc = SdC(
            numpy_rng=numpy_rng,
            n_ins=inDim,
            lbd = lbd, 
            beta = beta,
            input=x,
            hidden_layers_sizes= hidden_dim
        )
    else:
        try:
            with gzip.open(Init, 'rb') as f:
                saved_params = cPickle.load(f)['network']
            sdc = SdC(
                    numpy_rng=numpy_rng,
                    n_ins=inDim,
                    lbd = lbd, 
                    beta = beta,
                    input=x,
                    hidden_layers_sizes= hidden_dim,
                    Param_init = saved_params
                )
            print '... loading saved network succeeded'
        except IOError:
            print >> sys.stderr, ('Cannot find the specified saved network, using random initializations.')
            sdc = SdC(
                    numpy_rng=numpy_rng,
                    n_ins=inDim,
                    lbd = lbd, 
                    beta = beta,
                    input=x,
                    hidden_layers_sizes= hidden_dim
                )           
        
    #########################
    # PRETRAINING THE MODEL #
    #########################
    if pretraining_epochs == 0 or Init != '':
        print '... skipping pretraining'
    else:       
        print '... getting the pretraining functions'
        pretraining_fns = sdc.pretraining_functions(train_set_x=train_set_x,
                                                    batch_size=batch_size, mu = mu)

        print '... pre-training the model'
        start_time = timeit.default_timer()
        ## Pre-train layer-wise
        corruption_levels = 0*numpy.ones(len(hidden_dim), dtype = numpy.float32)
        
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
        
        network = [param.get_value() for param in sdc.params]    
        package = {'network': network}            
        with gzip.open('deepclus_'+str(nClass)+ '_pretrain.pkl.gz', 'wb') as f:
            cPickle.dump(package, f, protocol=cPickle.HIGHEST_PROTOCOL)

    ########################
    # FINETUNING THE MODEL #
    ########################
    
    
    km = KMeans(n_clusters = nClass)   
    
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

    hidden_zero = numpy.zeros_like(hidden_array)
    
    zeros_count = numpy.sum(numpy.equal(hidden_array, hidden_zero), axis = 0)       
    
#    # Do a k-means clusering to get center_array  
    km_idx = km.fit_predict(hidden_array)
    centers = km.cluster_centers_.astype(numpy.float32)     
    center_shared =  theano.shared(numpy.zeros((batch_size, hidden_dim[-1]) ,
                                                   dtype='float32'),
                                     borrow=True)
    nmi = metrics.normalized_mutual_info_score(label_true, km_idx)
    print >> sys.stderr, ('Initial NMI for deep clustering: %.2f' % (nmi))
    
    ari = metrics.adjusted_rand_score(label_true, km_idx)
    print >> sys.stderr, ('ARI for deep clustering: %.2f' % (ari))
    
    try:
        ac = acc(km_idx, label_true)
    except AssertionError:
        ac = 0
        print('Number of predicted cluster mismatch with ground truth.')
        
    print >> sys.stderr, ('ACC for deep clustering: %.2f' % (ac))                              
    lr_shared = theano.shared(numpy.asarray(finetune_lr,
                                                   dtype='float32'),
                                     borrow=True)

    print '... getting the finetuning functions'   
       
    train_fn = sdc.build_finetune_functions(
        datasets=datasets,
        centers=center_shared ,
        batch_size=batch_size,
        mu = mu,
        learning_rate=lr_shared
    )

    print '... finetunning the model'

    start_time = timeit.default_timer()
    done_looping = False
    epoch = 0
    
    res_metrics = numpy.zeros((training_epochs/5 + 1, 3), dtype = numpy.float32)
    res_metrics[0] = numpy.array([nmi, ari, ac])
    
    count = 100*numpy.ones(nClass, dtype = numpy.int)
    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1    
        c = [] # total cost
        d = [] # cost of reconstruction    
        e = [] # cost of clustering 
        f = [] # learning_rate
        g = []
        # count the number of assigned  data sample
        # perform random initialization of centroid if empty cluster happens
        count_samples = numpy.zeros((nClass)) 
        for minibatch_index in xrange(n_train_batches):
            # calculate the stepsize
            iter = (epoch - 1) * n_train_batches + minibatch_index 
            lr_shared.set_value( numpy.float32(finetune_lr) )
            center_shared.set_value(centers[km_idx[minibatch_index * batch_size: (minibatch_index +1 ) * batch_size]])
#            lr_shared.set_value( numpy.float32(finetune_lr/numpy.sqrt(epoch)) )
            cost = train_fn(minibatch_index)
            hidden_val = out_sdc(minibatch_index) # get the hidden value, to update KM
            # Perform mini-batch KM
            temp_idx, centers, count = batch_km(hidden_val, centers, count)
#            for i in range(nClass):
#                count_samples[i] += temp_idx.shape[0] - numpy.count_nonzero(temp_idx - i)             
#            center_shared.set_value(numpy.float32(temp_center))
            km_idx[minibatch_index * batch_size: (minibatch_index +1 ) * batch_size] = temp_idx
            c.append(cost[0])
            d.append(cost[1])
            e.append(cost[2])
            f.append(cost[3])
            g.append(cost[4])

        # check if empty cluster happen, if it does random initialize it
#        for i in range(nClass):
#            if count_samples[i] == 0:
#                rand_idx = numpy.random.randint(low = 0, high = n_train_samples)
#                # modify the centroid
#                centers[i] = out_single(rand_idx)                
        
        print 'Fine-tuning epoch %d ++++ \n' % (epoch), 
        print ('Total cost: %.5f, '%(numpy.mean(c)) + 'Reconstruction: %.5f, ' %(numpy.mean(d)) 
            + "Clustering: %.5f, " %(numpy.mean(e)) )
#        print 'Learning rate: %.6f' %numpy.mean(f)
        
        # half the learning rate every 5 epochs
        if epoch % 10 == 0 and diminishing == True:
            finetune_lr /= 2
            
#         evaluate the clustering performance every 5 epoches      
        if epoch % 5 == 0:            
            nmi = metrics.normalized_mutual_info_score(label_true, km_idx)                
            ari = metrics.adjusted_rand_score(label_true, km_idx)                
            try:
                ac = acc(km_idx, label_true)
            except AssertionError:
                ac = 0
                print('Number of predicted cluster mismatch with ground truth.')    
            res_metrics[epoch/5] = numpy.array([nmi, ari, ac])

    # get the hidden values, to make a plot
    hidden_val = [] 
    for batch_index in xrange(n_train_batches):
         hidden_val.append(out_sdc(batch_index))    
    hidden_array  = numpy.asarray(hidden_val)
    hidden_size = hidden_array.shape        
    hidden_array = numpy.reshape(hidden_array, (hidden_size[0] * hidden_size[1], hidden_size[2] ))
        
    err = numpy.mean(d)
    print >> sys.stderr, ('Average squared 2-D reconstruction error: %.4f' %err)
    end_time = timeit.default_timer()
    ypred = km_idx
    
    nmi = metrics.normalized_mutual_info_score(label_true, ypred)
    print >> sys.stderr, ('NMI for deep clustering: %.2f' % (nmi))

    ari = metrics.adjusted_rand_score(label_true, ypred)
    print >> sys.stderr, ('ARI for deep clustering: %.2f' % (ari))
    
    try:
        ac = acc(ypred, label_true)
    except AssertionError:
        ac = 0
        print('Number of predicted cluster mismatch with ground truth.')
        
    print >> sys.stderr, ('ACC for deep clustering: %.2f' % (ac))
    
    config = {'lbd': lbd,   
              'beta': beta,
              'pretraining_epochs': pretraining_epochs,
              'pretrain_lr': pretrain_lr, 
              'mu': mu,
              'finetune_lr': finetune_lr, 
              'training_epochs': training_epochs,
              'dataset': dataset, 
              'batch_size': batch_size, 
              'nClass': nClass, 
              'hidden_dim': hidden_dim}
    results = {'result': res_metrics}
    network = [param.get_value() for param in sdc.params]
    
    package = {'config': config,
               'results': results,
               'network': network}
    with gzip.open(save_file, 'wb') as f:          
        cPickle.dump(package, f, protocol=cPickle.HIGHEST_PROTOCOL)
        
    os.chdir(working_dir)    
    print >> sys.stderr, ('The training code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))   
    
    return res_metrics             

if __name__ == '__main__':      
    # run experiment with raw MNIST data
    K = 10
    filename = 'mnist_dcn.pkl.gz'
    path = '/home/bo/Data/MNIST/'
    
    trials = 1
    dataset = path+filename
    config = {'Init': '',
          'lbd': .05, 
          'beta': 1, 
          'output_dir': 'MNIST_results',
          'save_file': 'mnist_10.pkl.gz',
          'pretraining_epochs': 50,
          'pretrain_lr': .01, 
          'mu': 0.9,
          'finetune_lr': 0.05, 
          'training_epochs': 50,
          'dataset': dataset, 
          'batch_size': 128, 
          'nClass': K, 
          'hidden_dim': [2000, 1000, 500, 500, 250, 50],
          'diminishing': False}

    results = []
    for i in range(trials):         
        res_metrics = test_SdC(**config)   
        results.append(res_metrics)
        
    results_SAEKM = numpy.zeros((trials, 3)) 
    results_DCN   = numpy.zeros((trials, 3))

    N = config['training_epochs']/5
    for i in range(trials):
        results_SAEKM[i] = results[i][0]
        results_DCN[i] = results[i][N]
    SAEKM_mean = numpy.mean(results_SAEKM, axis = 0)    
    SAEKM_std  = numpy.std(results_SAEKM, axis = 0)    
    DCN_mean   = numpy.mean(results_DCN, axis = 0)
    DCN_std    = numpy.std(results_DCN, axis = 0)
    print >> sys.stderr, ('SAE+KM avg. NMI = {0:.2f}, ARI = {1:.2f}, ACC = {2:.2f}'.format(SAEKM_mean[0], 
                          SAEKM_mean[1], SAEKM_mean[2]) )    
    print >> sys.stderr, ('DCN    avg. NMI = {0:.2f}, ARI = {1:.2f}, ACC = {2:.2f}'.format(DCN_mean[0], 
                          DCN_mean[1], DCN_mean[2]) )   
    
    color  = ['b', 'g', 'r']
    marker = ['o', '+', '*']
    x = numpy.linspace(0, config['training_epochs'], num = config['training_epochs']/5 +1)    
    plt.figure(3)
    plt.xlabel('Epochs') 
    for i in range(3):
        y = res_metrics[:][:,i]        
        plt.plot(x, y, '-'+color[i]+marker[i], linewidth = 2)    
    plt.show()        
    plt.legend(['NMI', 'ARI', 'ACC'])        