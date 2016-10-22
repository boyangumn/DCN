# -*- coding: utf-8 -*-
"""
Created on Thu Aug 25 22:39:02 2016

This script is to pre-process RCV1-V2 dataset

@author: bo
"""

from sklearn.datasets import fetch_rcv1
import scipy.io as sio
import numpy
import gzip, cPickle
import os

target_dir = '/home/bo/Data/RCV1/Processed'
data_home = '/home/bo/Data'
#target_dir = '/project/sidir001/yang4173/Data/RCV1/Processed'
#data_home = '/project/sidir001/yang4173/Data'

cwd = os.getcwd()
data = fetch_rcv1(data_home = data_home, download_if_missing = True)
names = data.target_names

ind = numpy.full(len(names), False, dtype = bool)
f = open(data_home + '/RCV1/rcv1.topics.hier.orig.txt', 'r')
count = 0
for i in range(len(names) + 1):
    s = f.readline()
    if s[9:12] == 'CAT':
        ind[i - 1] = True
        count = count + 1
f.close()

labels = data.target[:][:, ind].copy()
labels = labels.toarray()
t = labels.sum(axis = 1, keepdims = False)
single_docs = numpy.where(t == 1)[0]

# keep only the documents with single label
labels = labels[single_docs]
docs = data.data[single_docs]

count = labels.sum(axis = 0, keepdims = False) 
ind = numpy.argsort(count)
ind = ind[::-1]
sort_count = count[ind]

# Creat subset of data top-4, top-8, ... top-20
# The first cluster is removed, due to its huge size
os.chdir(target_dir)

# save the whole training set
train_x = docs
train_y = labels.argmax(axis = 1)
sio.savemat('rcv_whole',{'train_x': train_x, 'train_y':train_y})

#

#for i in range(5):
i = 0
t = (i+1)*4+1
ind_sub = labels[:][:, ind[1:t]]
doc_ind = numpy.logical_or.reduce(ind_sub, axis = 1)

train_x = docs[doc_ind]
 # pick the most-frequent 2000 features
frequency = numpy.squeeze(numpy.asarray(train_x.sum(axis = 0)))
fre_ind = numpy.argsort(frequency)
fre_ind = fre_ind[::-1]
train_x = train_x[:][:, fre_ind[0:3000]]

train_y_mat = labels[doc_ind]
train_y = train_y_mat.argmax(axis = 1)

with gzip.open('data-'+str(44)+'.pkl.gz', 'wb') as f:
    cPickle.dump((train_x, train_y), f, protocol = 2)

sio.savemat('data-'+str(i), {'train_x': train_x, 'train_y': train_y})

z = 1
    
os.chdir(cwd)
        