# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 22:15:30 2016

retrive the saved results

@author: bo
"""

import cPickle, gzip

saved_file = 'deepclus_2_clusters.pkl.gz'
with gzip.open(saved_file, 'rb') as f:
    content = cPickle.load(f)

