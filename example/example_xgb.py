# -*- coding: utf-8 -*-
"""
@author: Satoshi Hara
"""

import sys
sys.path.append('../')

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

# train xgboost
num_round = 50
dtrain = xgb.DMatrix(Xtr, label=ytr)
param = {'max_depth':4, 'eta':0.3, 'silent':1, 'objective':'binary:logistic'}
bst = xgb.train(param, dtrain, num_round)

# output xgb model as text
bst.dump_model('xgbmodel.txt')

# fit simplified model
Kmax = 10
splitter = DefragModel.parseXGBtrees('./xgbmodel.txt') # parse XGB model into the array of (feature index, threshold)
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
