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
os.system('rscript regforest.R')

# fit simplified model
Kmax = 10
splitter = DefragModel.parseRtrees('./forest/') # parse R trees in ./forest/ into the array of (feature index, threshold)
mdl = DefragModel(modeltype='regression', maxitr=100, tol=1e-6, restart=20, verbose=0)
mdl.fit(ytr, Xtr, splitter, Kmax, fittype='FAB')

# results
score, cover = mdl.evaluate(yte, Xte)
print()
print('<< defragTrees >>')
print('----- Evaluated Results -----')
print('Test Error = %f' % (score,))
print('Test Coverage = %f' % (cover,))
print()
print('----- Found Rules -----')
print(mdl)
