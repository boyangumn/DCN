# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 21:56:33 2016

Perform experiment on Raw-MNIST data

@author: bo
"""


import gzip 
import cPickle
import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, metrics
from multi_layer_km import test_SdC
from cluster_acc import acc

K = 10
trials = 10

filename = 'mnist_dcn.pkl.gz'
path = '/home/bo/Data/MNIST/'
dataset = path+filename

## perform KM

with gzip.open(dataset, 'rb') as f:
        train_x, train_y = cPickle.load(f)
km_model = KMeans(n_clusters = K, n_init = 1) 
results_KM = np.zeros((trials, 3))
for i in range(trials):
    ypred = km_model.fit_predict(train_x)
    nmi = metrics.adjusted_mutual_info_score(train_y, ypred)
    ari = metrics.adjusted_rand_score(train_y, ypred)
    ac  = acc(ypred, train_y)
    results_KM[i] = np.array([nmi, ari, ac])

KM_mean = np.mean(results_KM, axis = 0)
KM_std  = np.std(results_KM, axis = 0)   

# perform DCN
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

color  = ['b', 'g', 'r']
marker = ['o', '+', '*']
x = np.linspace(0, config['training_epochs'], num = config['training_epochs']/5 +1)    
plt.figure(3)
plt.xlabel('Epochs') 
for i in range(3):
    y = res_metrics[:][:,i]        
    plt.plot(x, y, '-'+color[i]+marker[i], linewidth = 2)    
plt.show()        
plt.legend(['NMI', 'ARI', 'ACC'])

print >> sys.stderr, ('KM avg. NMI = {0:.2f}, ARI = {1:.2f}, ACC = {2:.2f}'.format(KM_mean[0], 
                      KM_mean[1], KM_mean[2]) )    
print >> sys.stderr, ('SAE+KM avg. NMI = {0:.2f}, ARI = {1:.2f}, ACC = {2:.2f}'.format(SAEKM_mean[0], 
                      SAEKM_mean[1], SAEKM_mean[2]) )    
print >> sys.stderr, ('DCN    avg. NMI = {0:.2f}, ARI = {1:.2f}, ACC = {2:.2f}'.format(DCN_mean[0], 
                      DCN_mean[1], DCN_mean[2]) )   
