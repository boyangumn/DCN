This is an introduction of the code developed for the Deep Clustering Network (DCN). Please direct your emails to 

Bo Yang, yang4173@umn.edu

if you have troubles running the code, or find any bugs. 

Here is the paper: arxiv: https://arxiv.org/pdf/1610.04794v1.pdf
Bo Yang, Xiao Fu, Nicholas D. Sidiropoulos and Mingyi Hong "Towards K-means-friendly Spaces: Simultaneous Deep Learning and Clustering"

==============================================
Main files 

run_rcv1.py       : Script to reproduce our results on RCV1 dataset (Table 1).
run_20News.py     : Script to reproduce our results on 20Newsgroup dataset (Table 2).
run_raw_mnist.py  : Script to reproduce our results on raw-MNIST dataset (Table 3).
run_pre_mnist.py  : Script to reproduce our results on pre-processed MNIST dataset (Table 4).
run_pendigits.py  : Script to reproduce our results on Pendigits dataset (Table 5).
multi_layer_km.py : Main file for defining the network, as well as various utility functions.

--
More documentations can be found inside each of the above files.
--
There are some additional python source files in the repository, which were developed for trying out various ideas. They are kept for possible future use, but are less documented (unfortunately...).  

==============================================
Data preparation

The data file should be named like 'something.pkl.gz', i.e., it should be pickled and compressed by gzip, using python code as follow:

"""
with gzip.open('something.pkl.gz', 'wb') as f:
    cPickle.dump([train_x, train_y], f, protocol = 0)
"""
where train_x and train_y are numpy ndarray with shape
train_x: (n_samples, n_features)
train_y: (n_samples, )

The path of data files should be made available to the programs by modifying the 'path' local variable in each of the 'run_' python source files.

==============================================
Dependencies

Theano   
scikit-learn
numpy
scipy
matplotlib
Theanon deep learning tutorial code (can be download from http://deeplearning.net/tutorial/).



