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

07/11/2016 Changed to use RBM as pretraining network. Changed dA2 class initialization procedure, to allow for external initialization;
            Changed initialization procedure of SdC class. Create a new class HiddenLayer2, to allow for initialization



"""

import os
import sys
import timeit
import scipy.io as sio
import copy

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
from RBMs_init import RBMs_init
#from deepclustering import load_data
from mlp import HiddenLayer
from multi_layer_km import SdC
from multi_layer_km import batch_km

try:
    import PIL.Image as Image
except ImportError:
    import Image
    
#theano.config.compute_test_value = 'warn'

# class dA2 inherited from dA, with loss function modified to norm-square loss
class dA2(dA):
    # overload the original function in dA class
    def __init__(
            self,
            numpy_rng,
            theano_rng=None,
            input=None,
            n_visible=784,
            n_hidden=500,
            W=None,
            bhid=None,
            bvis=None
        ):
        """
        Initialize the dA class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the corruption level. The
        constructor also receives symbolic variables for the input, weights and
        bias. Such a symbolic variables are useful when, for example the input
        is the result of some computations, or when weights are shared between
        the dA and an MLP layer. When dealing with SdAs this always happens,
        the dA on layer 2 gets as input the output of the dA on layer 1,
        and the weights of the dA are used in the second stage of training
        to construct an MLP.

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


        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if W is None:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
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

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]
    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
#        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        
        L = T.sum(T.pow(self.x - z, 2), axis = 1)                
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)        
#        
## class HiddenLayer2 
class HiddenLayer2(HiddenLayer):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input
        # end-snippet-1

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
        else:
            W_values = W

        W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        else:
            b_values = b;
            
        b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )
        # parameters of the model
        self.params = [self.W, self.b]        

# class SdC, main class for deep-clustering     
class SdC2(SdC):
    
    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input = None,
        n_ins=784,
        lbd = 1,
        hidden_layers_sizes=[1000, 200, 10],
        corruption_levels=[0, 0, 0],
        Param_init = None
    ):
        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        self.lbd = lbd
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
#                layer_input = self.sigmoid_layers[-1].output
                layer_input = self.dA_layers[-1].get_hidden_values(self.dA_layers[-1].x)
    
            # Construct a deep_clus layer, collect them together in the dc_layers list
    
            sigmoid_layer = HiddenLayer2(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        W = Param_init[3*i],
                                        b = Param_init[3*i + 1],
                                        activation=T.nnet.sigmoid)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)
            # its arguably a philosophical question...
            # but we are going to only declare that the parameters of the
            # sigmoid_layers are parameters of the StackedDAA
            # the visible biases in the dA are parameters of those
            # dA, but not the SdA
            
            
            # using the dA2 objects, instead of dA.
            # dA2 uses norm-square loss function
            dA_layer = dA2(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W = Param_init[3*i],
                          bhid = Param_init[3*i + 1],
                          bvis = Param_init[3*i+2])
            self.dA_layers.append(dA_layer)         
            
            self.params.extend(dA_layer.params)
            
            delta_i = (theano.shared(value = numpy.zeros((input_size, hidden_layers_sizes[i]), dtype = numpy.float32), borrow=True), 
                       theano.shared(value = numpy.zeros(hidden_layers_sizes[i],  dtype = numpy.float32), borrow = True ),
                        theano.shared(value = numpy.zeros(input_size,  dtype = numpy.float32), borrow = True ) )
            self.delta.extend(delta_i)
    def finetune_cost_updates(self, prototypes_y, prototypes_r, mu, learning_rate):
        """ This function computes the cost and the updates ."""

        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, withd one entry per
        #        example in minibatch
        # Using least-squares loss for both clustering 
        # No reconstruction cost in this version
        network_output = self.get_output()        
        L = T.sum(T.maximum(0, 1 + T.sum(prototypes_r * network_output, axis = 1) 
                - T.sum(prototypes_y * network_output, axis = 1) ), axis = 0)        
        
#        temp = T.pow(center - network_output, 2)    
#        
#        L =  T.sum(temp, axis=1) 
        # Add the network reconstruction error 
        z = self.get_network_reconst()
        reconst_err = T.sum(T.pow(self.x - z, 2), axis = 1)     
#        reconst_err = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        
        L = L + self.lbd*reconst_err
        
        cost1 = T.mean(L)
        cost2 = self.lbd*T.mean(reconst_err)  
        cost3 = cost1 - cost2

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost1, self.params)
        # generate the list of updates
        updates = []
        for param, delta, gparam in zip(self.params, self.delta, gparams):
            updates.append( (delta, mu*delta - learning_rate * gparam) )
            updates.append( (param, param + mu*mu*delta - (1+mu)*learning_rate*gparam ))
            
        return ((cost1, cost2, cost3, learning_rate), updates)
    
    def build_finetune_functions(self, datasets, prototypes_y, prototypes_r, batch_size, mu, learning_rate):
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

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        
        ONLY TRAINGING IS IMPLEMENTED, VALIDATION AND TESTING TO BE ADDED...
        '''

        (train_set_x, train_set_y) = datasets[0]
#        (valid_set_x, valid_set_y) = datasets[1]
#        (test_set_x, test_set_y)   = datasets[2]
        
#        center= T.matrix('center')
        
        # compute number of minibatches for training, validation and testing
#        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
#        n_valid_batches /= batch_size
#        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
#        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        cost, updates = self.finetune_cost_updates(
        prototypes_y,
        prototypes_r,
        mu,
        learning_rate=learning_rate
        )
        
        train_fn = theano.function(
            inputs=[index],
            outputs= cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )
        return train_fn       

def batch_mmc(prototypes, data, eta, lr):
    """
    Implementing MMC clustering
    
    Reference: Zhangyang Wang et.al, 
    A Joint Optimization Framework of Sparse Coding and Discriminative Clustering, IJCAI 2015
    
    Assuming ROWs of prototypes and data, K prototypes, N data points
    
    """
    K = prototypes.shape[0] # number of clusters
    N = data.shape[0]       # number of data points
    # first column of assignment are the y_i's, second column are the r_i's
    new_assignment = numpy.zeros((N, 2), dtype = numpy.int) 
    new_prototypes = numpy.zeros_like(prototypes, dtype = numpy.float32)
    # get the inner product
    prod = numpy.dot(prototypes, data.T)
    # sorting the rows, in ascending order
    ind = numpy.argsort(prod, axis = 0)  
    
    new_assignment[:][:, 0] = ind[-1].T
    new_assignment[:][:, 1] = ind[-2].T
    
    for k in range(K):
#        grad = numpy.zeros_like(prototypes[0])
        grad = eta*prototypes[k]
        for n in range(N):
            if new_assignment[n, 0] == k:
                grad = grad - data[n]
            elif new_assignment[n,1] == k:
                grad = grad + data[n]
            else:
                continue
        
        new_prototypes[k] = prototypes[k] - lr*grad
    
    return new_assignment, new_prototypes
    
    
def load_all_data(dataset):
    ''' Loads the dataset

    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    '''

    #############
    # LOAD DATA #
    #############

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print 'Downloading data from %s' % origin
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    # Load the dataset
    f = gzip.open(dataset, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
# K-means clustering 
    
    train_x = numpy.concatenate((train_set[0], test_set[0], valid_set[0]), axis = 0)   
    train_y = numpy.concatenate((train_set[1], test_set[1], valid_set[1]), axis = 0)   
    train_x_reduced = train_x[:][:, 49:650]
    
#    f = gzip.open('MNIST_array', 'wb')
#    cPickle.dump([train_x, train_y, train_x_reduced], f, protocol=2)
#    f.close()    
#    S = numpy.linalg.svd(train_x, compute_uv = 1)    
    
## detecting the required rank to preserve 95% energy, the result is 427    
#    aa = numpy.cumsum(S)
#    bb = numpy.sum(S)
#    for j in range(len(aa)):
#        if aa[j]/bb > 0.95:
#            break
    
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
            
    data_x, data_y = shared_dataset((train_x_reduced, train_y))
    
    
#    test_set_x, test_set_y = shared_dataset(test_set)
#    valid_set_x, valid_set_y = shared_dataset(valid_set)
#    train_set_x, train_set_y = shared_dataset(train_set)
#
    # The value 0 won't be used, just use as a placeholder
    rval = [(data_x, data_y), 0, 0]
    return rval
           
def test_SdC(lbd = .01, finetune_lr= .005, mu = 0.9, pretraining_epochs=50,
             pretrain_lr=.001, mmc_eta = 0.001, mmc_lr = 0.01, training_epochs=150, init = 'rbm_init_30',
             dataset='mnist.pkl.gz', batch_size=20, nClass = 10, hidden_dim = [10, 10]):
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

    datasets = load_all_data(dataset)  

    train_set_x, train_set_y = datasets[0]
#    valid_set_x, valid_set_y = datasets[1]
#    test_set_x,  test_set_y  = datasets[2]
    
    inDim = train_set_x.get_value().shape[1]
    label_true = numpy.int32(train_set_y.get_value(borrow=True))
    
    index = T.lscalar() 
    x = T.matrix('x')
    
#    load the save model
#    f = open('RBMs_init_1000.save', 'rb')
#    param_init = cPickle.load(f)
#    
#    for i in range(len(param_init)):
#        param_init[i] = param_init[i]/4
        
#   load the network trained with Hinton's code    
    var_names = ['vishid', 'hidrecbiases', 'visbiases',
    'hidpen', 'penrecbiases', 'hidgenbiases',
    'hidpen2', 'penrecbiases2', 'hidgenbiases2',
    'hidtop', 'toprecbiases', 'topgenbiases']
    mat_init = sio.loadmat(init) 
    
    param_init = []
    for i in range(12):
        param_init.append( numpy.squeeze( numpy.float32(mat_init[var_names[i]]) ) )
    
    numpy_rng = numpy.random.RandomState(125)
#    RBMs = RBMs_init(
#        numpy_rng=numpy_rng,
#        theano_rng=None,
#        n_ins=inDim,
#        hidden_layers_sizes = hidden_dim, n_outs=10 ,
#    )
#    RBMs.params = param_init
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size

    # numpy random generator
    # start-snippet-3
    print '... building the model'
    # construct the stacked denoising autoencoder class
    sdc = SdC2(
        numpy_rng=numpy_rng,
        n_ins=inDim,
        lbd = lbd, 
        input=x,
        hidden_layers_sizes= hidden_dim,
        Param_init = param_init
    )
    # end-snippet-3 start-snippet-4
    #########################
    # Load the initialization #
    #########################

#    print '... loading the saved initialization network'
#    start_time = timeit.default_timer()
#
#    for i in xrange(sdc.n_layers):
#        sdc.sigmoid_layers[i].W = RBMs.params[3*i]  
#        sdc.sigmoid_layers[i].b = RBMs.params[3*i + 1] 
#        
#        sdc.dA_layers[i].W = RBMs.params[3*i]  
#        sdc.dA_layers[i].b = RBMs.params[3*i + 1] 
#        sdc.dA_layers[i].b_prime = RBMs.params[3*i + 2] 
        

    # end-snippet-4
    ########################
    # FINETUNING THE MODEL #
    ########################
    
    km = MiniBatchKMeans(n_clusters = nClass, batch_size=10000)   
    
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
      
#    # use the true labels to get initial cluster centers
#    centers = numpy.zeros((nClass, hidden_size[2]))
#    
#    for i in xrange(nClass):
#        temp = hidden_array[label_true == i]        
#        centers[i] = numpy.mean(temp, axis = 0)      
#    
#    center_array = centers[label_true]
#    # Do a k-means clusering to get center_array  
    km_idx = km.fit_predict(hidden_array)
    
    # initializing the prototypes
    prototypes = numpy.zeros((nClass, hidden_dim[-1]), dtype = numpy.float32)
    for k in range(nClass):
        data_k = hidden_array[km_idx == k]
        U, S, V = numpy.linalg.svd(data_k, full_matrices = 0)
        prototypes[k] = V[:][:,0].T   
#    km_idx = copy.deepcopy(label_true)
    
#    prototypes = numpy.random.randn(nClass, hidden_dim[-1])
    nmi_dc = metrics.adjusted_mutual_info_score(label_true, km_idx)
    print >> sys.stderr, ('Initial NMI for deep clustering: %.2f' % (nmi_dc))
    
#    centers = numpy.zeros((nClass, hidden_size[2]), dtype = numpy.float32)        
#    
#    for i in xrange(nClass):
#        temp = hidden_array[km_idx == i]        
#        centers[i] = numpy.mean(temp, axis = 0)
##    center_array = km.cluster_centers_[[km.labels_]]             
#    center_shared =  theano.shared(numpy.zeros((batch_size, hidden_dim[-1]) ,
#                                                   dtype='float32'),
#                                     borrow=True)
                                     
    prototypes_y_shared =  theano.shared(numpy.zeros((batch_size, hidden_dim[-1]) ,
                                                   dtype='float32'),
                                     borrow=True)
    prototypes_r_shared =  theano.shared(numpy.zeros((batch_size, hidden_dim[-1]) ,
                                                   dtype='float32'),
                                     borrow=True)
    
    lr_shared = theano.shared(numpy.asarray(finetune_lr,
                                                   dtype='float32'),
                                     borrow=True)

    print '... getting the finetuning functions'   
       
    train_fn = sdc.build_finetune_functions(
        datasets=datasets,
        prototypes_y = prototypes_y_shared,
        prototypes_r = prototypes_r_shared,        
        batch_size=batch_size,
        mu = mu, 
        learning_rate=lr_shared,
    )

    print '... finetunning the model'
    # early-stopping parameters

    start_time = timeit.default_timer()
    done_looping = False
    epoch = 0
    
#    count = 100*numpy.ones(nClass)
    assignment = numpy.zeros((train_set_y.get_value().shape[0], 2), dtype = numpy.int)
    while (epoch < training_epochs) and (not done_looping):
        epoch = epoch + 1    
        c = [] # total cost
        d = [] # cost of reconstruction    
        e = [] # cost of clustering 
        f = [] # learning_rate
        g = []
        for minibatch_index in range(n_train_batches):
            # calculate the stepsize
            iter = (epoch - 1) * n_train_batches + minibatch_index 
            lr_shared.set_value( numpy.float32(finetune_lr/numpy.sqrt(epoch)) )
#            center_shared.set_value(centers[km_idx[minibatch_index * batch_size: (minibatch_index +1 ) * batch_size]])
            hidden_val = out_sdc(minibatch_index) # get the hidden value, to update KM
            # Perform mini-batch MMC
            temp_assignment, prototypes = batch_mmc(prototypes, hidden_val, mmc_eta, mmc_lr)
            assignment[minibatch_index * batch_size: (minibatch_index +1 ) * batch_size] = temp_assignment
            
            prototypes_y_shared.set_value(prototypes[temp_assignment[:][:,0]])
            prototypes_r_shared.set_value(prototypes[temp_assignment[:][:,1]])            
            
            cost = train_fn(minibatch_index)
#            hidden_val = out_sdc(minibatch_index) # get the hidden value, to update KM
#            # Perform mini-batch MMC
#            temp_assignment, prototypes = batch_mmc(prototypes, hidden_val, mmc_eta, mmc_lr)
#            assignment[minibatch_index * batch_size: (minibatch_index +1 ) * batch_size] = temp_assignment
            
#            temp_idx, centers, count = batch_km(hidden_val, centers, count)
#            center_shared.set_value(numpy.float32(temp_center))
#            km_idx[minibatch_index * batch_size: (minibatch_index +1 ) * batch_size] = temp_idx
            
            bb = sdc.dA_layers[0].W.get_value()
            c.append(cost[0])
            d.append(cost[1])
            e.append(cost[2])
            f.append(cost[3])
#            gg = cost[4]
#            g.append(cost[4])
            
            
            
        # Do a k-means clusering to get center_array    
#        hidden_val = [] 
#        for batch_index in xrange(n_train_batches):
#             hidden_val.append(out_sdc(batch_index))
#        
#        hidden_array  = numpy.asarray(hidden_val)
#        hidden_size = hidden_array.shape        
#        hidden_array = numpy.reshape(hidden_array, (hidden_size[0] * hidden_size[1], hidden_size[2] ))
#        km.fit(hidden_array)
#        center_array = km.cluster_centers_[[km.labels_]]   
#        center_shared.set_value(numpy.asarray(center_array, dtype='float32'))          
#        center_shared =  theano.shared(numpy.asarray(center_array ,
#                                                       dtype='float32'),
#                                         borrow=True)   
        print 'Fine-tuning epoch %d ++++ \n' % (epoch), 
        print ('Total cost: %.5f, '%(numpy.mean(c)) + 'Reconstruction: %.5f, ' %(numpy.mean(d)) 
            + "Clustering: %.5f, " %(numpy.mean(e)) )
#        print 'Learning rate: %.6f' %numpy.mean(f)
    
    # get the hidden values, to make a plot
    ypred = assignment[:][:, 0]
    hidden_val = [] 
    for batch_index in xrange(n_train_batches):
         hidden_val.append(out_sdc(batch_index))    
    hidden_array  = numpy.asarray(hidden_val)
    hidden_size = hidden_array.shape        
    hidden_array = numpy.reshape(hidden_array, (hidden_size[0] * hidden_size[1], hidden_size[2] ))
    
    err = numpy.mean(d)
    print >> sys.stderr, ('Average squared 2-D reconstruction error: %.4f' %err)
    end_time = timeit.default_timer()
#    ypred = km.predict(hidden_array)   
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

    a = 0
  
#    if dataset == 'toy.pkl.gz':
#        x = train_set_x.get_value()[:, 0]
#        y = train_set_x.get_value()[:, 1]
#        
#        # using resulted label, and the original data
#        pred_label = ypred[0:1999]
#        for i in xrange(nClass):
#            idx_x = x[numpy.nonzero( pred_label == i)]
#            idx_y = y[numpy.nonzero( pred_label == i)]   
#            plt.figure(4)
#            plt.scatter(idx_x, idx_y, s = 70, c = color[i], marker = marker[i], label = '%s'%i)
#        
#        plt.legend()
#        plt.show() 
           
    
# hidden_dim = [1000, 1000, 1000]
# hidden_dim = [1000, 500, 250, 30]
if __name__ == '__main__':      
    params = {'lbd': 0.05, 
              'finetune_lr': 0.005, 
              'mu': 0.9,
              'pretraining_epochs': 50,
              'pretrain_lr': .1, 
              'mmc_eta': .001,
              'mmc_lr': 1e-6,
              'training_epochs': 10,
              'init': 'rbm_reduced_30.mat',
              'dataset': 'mnist.pkl.gz', 
              'batch_size': 50, 
              'nClass': 10, 
              'hidden_dim': [1000, 500, 250, 30]}
             
    test_SdC(**params)
             
 
"""
## best NMI on MNIST 0.55   

params = {'lbd': 0.001, 
              'finetune_lr': 0.001, 
              'mu': 0.9,
              'pretraining_epochs': 50,
              'pretrain_lr': .1, 
              'training_epochs': 100,
              'dataset': 'mnist.pkl.gz', 
              'batch_size': 20, 
              'nClass': 10, 
              'hidden_dim': [1000, 500, 250, 12]}
initialize with finetuned-rbms
Minibatch Kmeans, with count initialized as 
count = 100*numpy.ones(nClass)
              
"""
        