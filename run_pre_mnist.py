# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 08:50:00 2016

Experiments on pre-processed MNIST

@author: bo
"""


import sys
import gzip 
import cPickle
import numpy as np

from sklearn import metrics
from sklearn.cluster import KMeans
from multi_layer_km import test_SdC
from cluster_acc import acc

K = 10
trials = 10

filename = 'pre_mnist.pkl.gz'
path = '/home/bo/Data/MNIST/'
dataset = path+filename

with gzip.open(dataset, 'rb') as f:
    train_x, train_y = cPickle.load(f)

np.random.seed(seed = 1)
# perform KM
km_model = KMeans(n_clusters = K, n_init = 1) 
results_KM = np.zeros((trials, 3))
for i in range(trials):
    ypred = km_model.fit_predict(train_x)
    nmi = metrics.normalized_mutual_info_score(train_y, ypred)
    ari = metrics.adjusted_rand_score(train_y, ypred)
    ac  = acc(ypred, train_y)
    results_KM[i] = np.array([nmi, ari, ac])

KM_mean = np.mean(results_KM, axis = 0)
KM_std  = np.std(results_KM, axis = 0)  

# perform DCN
config = {'Init': '',
          'lbd': 0.1, 
          'beta': 1, 
          'output_dir': 'MNIST_results',
          'save_file': 'mnist_ssc.pkl.gz',
          'pretraining_epochs': 10,
          'pretrain_lr': 0.01, 
          'mu': 0.9,
          'finetune_lr': 0.01, 
          'training_epochs': 50,
          'dataset': dataset, 
          'batch_size': 20, 
          'nClass': K, 
          'hidden_dim': [50, 20, 5],
          'diminishing': False}

results = []
for i in range(trials):         
    res_metrics = test_SdC(**config)   
    results.append(res_metrics)
    
results_SAEKM = np.zeros((trials, 3)) 
results_DCN   = np.zeros((trials, 3))

N = config['training_epochs']/5
for i in range(trials):
    results_SAEKM[i] = results[i][0]
    results_DCN[i] = results[i][N]
SAEKM_mean = np.mean(results_SAEKM, axis = 0)    
SAEKM_std  = np.std(results_SAEKM, axis = 0)    
DCN_mean   = np.mean(results_DCN, axis = 0)
DCN_std    = np.std(results_DCN, axis = 0)

print >> sys.stderr, ('KM avg. NMI = {0:.2f}, ARI = {1:.2f}, ACC = {2:.2f}'.format(KM_mean[0], 
                      KM_mean[1], KM_mean[2]) )                           
print >> sys.stderr, ('SAE+KM avg. NMI = {0:.2f}, ARI = {1:.2f}, ACC = {2:.2f}'.format(SAEKM_mean[0], 
                      SAEKM_mean[1], SAEKM_mean[2]) )    
print >> sys.stderr, ('DCN    avg. NMI = {0:.2f}, ARI = {1:.2f}, ACC = {2:.2f}'.format(DCN_mean[0], 
                      DCN_mean[1], DCN_mean[2]) )   

