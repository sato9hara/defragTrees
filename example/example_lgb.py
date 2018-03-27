# -*- coding: utf-8 -*-
"""
@author: Satoshi Hara
"""

import sys
sys.path.append('../')

import numpy as np
import lightgbm as lgb
from defragTrees import DefragModel

# load data
Ztr = np.loadtxt('./train.csv', delimiter=',')
Zte = np.loadtxt('./test.csv', delimiter=',')
Xtr = Ztr[:, :-1]
ytr = Ztr[:, -1].astype(np.int8)
Xte = Zte[:, :-1]
yte = Zte[:, -1].astype(np.int8)

# train LightGBM
gbm = lgb.LGBMClassifier(objective='binary',
                        num_leaves=31,
                        learning_rate=0.05,
                        n_estimators=50)
gbm.fit(Xtr, ytr,
        eval_set=[(Xte, yte)],
        eval_metric='binary_error',
        early_stopping_rounds=10)
z = gbm.predict(Xte, num_iteration=gbm.best_iteration_)

# output LightGBM model as text
gbm.booster_.save_model('lgbmodel.txt')

# fit simplified model
Kmax = 10
splitter = DefragModel.parseLGBtrees('./lgbmodel.txt') # parse LightGBM model into the array of (feature index, threshold)
mdl = DefragModel(modeltype='classification', maxitr=100, qitr=0, tol=1e-6, restart=20, verbose=0, L=5)
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
