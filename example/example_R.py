# -*- coding: utf-8 -*-
"""
@author: Satoshi Hara
"""

import sys
sys.path.append('../')

import os
import numpy as np
import xgboost as xgb
from defragTrees import DefragModel

# load data
Ztr = np.loadtxt('./train.csv', delimiter=',')
Zte = np.loadtxt('./test.csv', delimiter=',')
Xtr = Ztr[:, :-1]
ytr = Ztr[:, -1]
Xte = Zte[:, :-1]
yte = Zte[:, -1]

# train random foerst in R
os.system('Rscript clfforest.R')

# fit simplified model
Kmax = 10
splitter = DefragModel.parseRtrees('./forest/') # parse R trees in ./forest/ into the array of (feature index, threshold)
mdl = DefragModel(modeltype='classification', maxitr=100, qitr=0, tol=1e-6, restart=20, verbose=0)
mdl.fit(Xtr, ytr, splitter, Kmax, fittype='FAB')

# results
score, cover, coll = mdl.evaluate(Xte, yte)
print()
print('<< defragTrees >>')
print('----- Evaluated Results -----')
print('Test Error = %f' % (score,))
print('Test Coverage = %f' % (cover,))
print('Overlap = %f' % (coll,))
print()
print('----- Found Rules -----')
print(mdl)
