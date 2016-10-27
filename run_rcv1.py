# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 07:45:42 2016

Experiments on RCV1-v2

@author: bo
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from multi_layer_km import test_SdC, load_data
from cluster_acc import acc

trials = 1

# i = [0, 1, 2, 3, 4], corresponds to [4, 8, 12, 16, 20] clusters

i = 0
filename = 'data-'+str(i)+'.pkl.gz'
K = (i+1)*4
path = '/home/bo/Data/RCV1/Processed/'
dataset = path+filename


#np.random.seed(seed = 1)
## perform KM

train_x, train_y = load_data(dataset)
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

#   for RCV1    
config_1 = {'Init': '',
          'lbd': 0.1, 
          'beta': 1, 
          'output_dir': 'RCV_results',
          'save_file': 'rcv_10.pkl.gz',
          'pretraining_epochs': 50,
          'pretrain_lr': 0.01, 
          'mu': 0.9,
          'finetune_lr': 0.05, 
          'training_epochs': 50,
          'dataset': dataset, 
          'batch_size': 256, 
          'nClass': K, 
          'hidden_dim': [2000, 1000, 1000, 1000, 50],
          'diminishing': False}

config_2 = {'Init': '',
          'lbd': 0.1, 
          'beta': 1, 
          'output_dir': 'RCV_results',
          'save_file': 'rcv_10.pkl.gz',
          'pretraining_epochs': 50,
          'pretrain_lr': 0.01, 
          'mu': 0.9,
          'finetune_lr': 0.05, 
          'training_epochs': 50,
          'dataset': dataset, 
          'batch_size': 256, 
          'nClass': K, 
          'hidden_dim': [2000, 1000, 1000, 1000, 500, 500, 50],
          'diminishing': False}

results = []
for i in range(trials):     
    if K == 4 or K == 8:    
        # use configuration 1
        config = config_1 
    else:
        # use configuration 2
        config = config_2
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
#print >> sys.stderr, ('KM avg. NMI = {0:.2f}, ARI = {1:.2f}, ACC = {2:.2f}'.format(KM_mean[0], 
#                      KM_mean[1], KM_mean[2]) )                           
print >> sys.stderr, ('SAE+KM avg. NMI = {0:.2f}, ARI = {1:.2f}, ACC = {2:.2f}'.format(SAEKM_mean[0], 
                      SAEKM_mean[1], SAEKM_mean[2]) )    
print >> sys.stderr, ('DCN    avg. NMI = {0:.2f}, ARI = {1:.2f}, ACC = {2:.2f}'.format(DCN_mean[0], 
                      DCN_mean[1], DCN_mean[2]) )   
