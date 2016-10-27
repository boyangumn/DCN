# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 14:48:35 2016

Perform experiments with Pendigits

@author: bo
"""
import sys
import gzip 
import cPickle
import numpy as np

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.manifold import SpectralEmbedding
from multi_layer_km import test_SdC
from cluster_acc import acc

K = 10
trials = 10

filename = 'pendigits.pkl.gz'
path = '/home/bo/Data/Pendigits/'
dataset = path+filename

# perform KM

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
# Perform SC    
print('SC started...')
results_SC = np.zeros((trials, 3))
se_model = SpectralEmbedding(n_components=K, affinity='rbf', gamma = 0.1)
se_vec = se_model.fit_transform(train_x)
for i in range(trials):
    ypred = km_model.fit_predict(se_vec)
    nmi = metrics.adjusted_mutual_info_score(train_y, ypred)
    ari = metrics.adjusted_rand_score(train_y, ypred)
    ac  = acc(ypred, train_y)
    results_SC[i] = np.array([nmi, ari, ac])

SC_mean = np.mean(results_SC, axis = 0)
SC_std  = np.std(results_SC, axis = 0)   

# for PenDigits, perform DCN and SAE+KM
config = {'Init': '',
          'lbd': .5, 
          'beta': 1, 
          'output_dir': 'Pendigits',
          'save_file': 'pen_10.pkl.gz',
          'pretraining_epochs': 50,
          'pretrain_lr': 0.01, 
          'mu': 0.9,
          'finetune_lr': 0.01, 
          'training_epochs': 50,
          'dataset': dataset, 
          'batch_size': 20, 
          'nClass': K, 
          'hidden_dim': [50, 16, 10],
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
print >> sys.stderr, ('SC   avg. NMI = {0:.2f}, ARI = {1:.2f}, ACC = {2:.2f}'.format(SC_mean[0], 
                      SC_mean[1], SC_mean[2]) )   
print >> sys.stderr, ('SAE+KM avg. NMI = {0:.2f}, ARI = {1:.2f}, ACC = {2:.2f}'.format(SAEKM_mean[0], 
                      SAEKM_mean[1], SAEKM_mean[2]) )    
print >> sys.stderr, ('DCN    avg. NMI = {0:.2f}, ARI = {1:.2f}, ACC = {2:.2f}'.format(DCN_mean[0], 
                      DCN_mean[1], DCN_mean[2]) )   