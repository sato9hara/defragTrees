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
num_round = 20
dtrain = xgb.DMatrix(Xtr, label=ytr)
param = {'max_depth':4, 'eta':0.3, 'silent':1, 'objective':'reg:linear'}
bst = xgb.train(param, dtrain, num_round)

# output xgb model as text
bst.dump_model('xgbmodel.txt')

# fit simplified model
Kmax = 10
splitter = DefragModel.parseXGBtrees('./xgbmodel.txt') # parse XGB model into the array of (feature index, threshold)
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
