# -*- coding: utf-8 -*-
"""
@author: Satoshi Hara
"""

import sys
sys.path.append('../')

import numpy as np
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor
from defragTrees import DefragModel

# load data
Ztr = np.loadtxt('./train.csv', delimiter=',')
Zte = np.loadtxt('./test.csv', delimiter=',')
Xtr = Ztr[:, :-1]
ytr = Ztr[:, -1]
Xte = Zte[:, :-1]
yte = Zte[:, -1]

# train tree ensemble
forest = GradientBoostingRegressor(min_samples_leaf=10)
#forest = RandomForestRegressor(min_samples_leaf=10)
#forest = ExtraTreesRegressor(min_samples_leaf=10)
#forest = AdaBoostRegressor()
forest.fit(Xtr, ytr)

# fit simplified model
Kmax = 10
splitter = DefragModel.parseSLtrees(forest) # parse sklearn tree ensembles into the array of (feature index, threshold)
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
