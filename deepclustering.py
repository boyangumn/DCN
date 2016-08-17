# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 10:38:06 2016

@author: bo
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

#import matplotlib.pyplot as plt
from utils import tile_raster_images

#from logistic_sgd import LogisticRegression
#from mlp import HiddenLayer
from dA import dA

try:
    import PIL.Image as Image
except ImportError:
    import Image

class deep_clus (dA):
    """
        Inherit from dA class in denoising autoencoder example.
    """
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
        if not W:
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
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
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
    
    def get_cost_updates(self, center, corruption_level, learning_rate):
        """ This function computes the cost and the updates ."""

        tilde_x = self.get_corrupted_input(self.x, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        # Using least-squares loss for both clustering and reconstruction
        temp1 = T.pow(center - y, 2)
        temp2 = T.pow(self.x - z, 2)
        
        L =  T.sum(temp1  , axis=1) + T.sum(temp2  , axis=1) 
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
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


def load_data(dataset):
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
    
    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

# Prepare data
    
    
def deepclustering(learning_rate=0.1, training_epochs=15,
            dataset='mnist.pkl.gz',
            batch_size=20, output_folder='dA_plots'):
    """
    This demo is tested on MNIST

    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing
                          AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training

    :type dataset: string
    :param dataset: path to the pickled dataset

    """                
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]
    nHid = 2000
  # Load the saved dA object, to initialize our model

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    label_true = train_set_y.get_value(borrow=True)
    # start-snippet-2
    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    center= T.matrix('center')
    # end-snippet-2

    #if not os.path.isdir(output_folder):
    #    os.makedirs(output_folder)
   # os.chdir(output_folder)

    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################
    #Train a denosing autoencoder to initialize my own network, and provide latent representation for initializing clusteing

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))
    # Instancialize a dA class
    # To get the initial clustering information
    f = open('no_corruption.save', 'rb')
    no_corruption = cPickle.load(f)
    
    init_da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=28 * 28,
        n_hidden=nHid ,
    )
    init_da.params = no_corruption

    hid = init_da.get_hidden_values(x)
    hidden_da = theano.function(
        [index],
        outputs = hid,
        givens = {x: train_set_x[index * batch_size: (index + 1) * batch_size]}
    )
    # go through training epochs
    km = MiniBatchKMeans(n_clusters = 10,  batch_size=100)
    
    train_array = train_set_x.get_value()
    
    ypred = km.fit_predict(train_array)
    nmi_data = metrics.normalized_mutual_info_score(label_true, ypred)

    hidden_val = [] 
    for batch_index in xrange(n_train_batches):
         hidden_val.append(hidden_da(batch_index))
        
    hidden_array  = numpy.asarray(hidden_val)
    hidden_size = hidden_array.shape        
    hidden_array = numpy.reshape(hidden_array, (hidden_size[0] * hidden_size[1], hidden_size[2] ))
    # Do a k-means clusering to get center_array
    ypred = km.fit_predict(hidden_array)
    nmi_disjoint = metrics.normalized_mutual_info_score(label_true, ypred)
    
    center_array = km.cluster_centers_[[km.labels_]]         
    
    center_shared =  theano.shared(numpy.asarray(center_array ,
                                                   dtype='float32'),
                                     borrow=True)
    
    
    dc = deep_clus(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=28 * 28,
        n_hidden=nHid 
    )

    cost, updates = dc.get_cost_updates(
        center, 
        corruption_level=0.,
        learning_rate=learning_rate
    )
    #reconst = da.get_reconstructed_input(hidden)

    #  training a pure denoising autoencoder, without clustering, to get initial values to cluster

    train_dc = theano.function(
        inputs = [index],
        outputs = cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            center: center_shared[index * batch_size: (index + 1) * batch_size]
        }
    )

    start_time = timeit.default_timer()

    ############
    # TRAINING #
    ############

        
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            cost_batch = train_dc(batch_index)
            c.append(cost_batch)    
        
        print  'Training epoch %d, cost ' % epoch, numpy.mean(c)
        
        hidden_val = [] 
        for batch_index in xrange(n_train_batches):
            hidden_val.append( hidden_da(batch_index))
        
        hidden_array  = numpy.asarray(hidden_val)
        hidden_size = hidden_array.shape        
        hidden_array = numpy.reshape(hidden_array, (hidden_size[0] * hidden_size[1], hidden_size[2] ))
            
        km.init = km.cluster_centers_
        km.fit(hidden_array)
        
        center_array = km.cluster_centers_[[km.labels_]]         
        center_shared =  theano.shared(numpy.asarray(center_array ,
                                                   dtype='float32'),
                                     borrow=True)
        #        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = timeit.default_timer()
    ypred = km.predict(hidden_array)    
    
    nmi_dc = metrics.adjusted_mutual_info_score(label_true, ypred)
    print 'Normalized mutual info for data KMeans: ' ,   nmi_data
    print 'Normalized mutual info for disjoint clustering: ' ,   nmi_disjoint
    print 'Normalized mutual info for deep clustering: ' ,   nmi_dc
    
    training_time = (end_time - start_time)

    print >> sys.stderr, ('The no corruption code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((training_time) / 60.))
    image = Image.fromarray(
        tile_raster_images(X=dc.W.get_value(borrow=True).T,
                           img_shape=(28, 28), tile_shape=(10, 10),
                           tile_spacing=(1, 1)))
    image.save('filters_corruption_0.png')
    
   
    
if __name__ == '__main__':
    deepclustering()
    
