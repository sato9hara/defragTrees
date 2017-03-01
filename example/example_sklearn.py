# -*- coding: utf-8 -*-
"""
@author: Satoshi Hara
"""

import sys
sys.path.append('../')

import numpy as np
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier
from defragTrees import DefragModel

# load data
Ztr = np.loadtxt('./train.csv', delimiter=',')
Zte = np.loadtxt('./test.csv', delimiter=',')
Xtr = Ztr[:, :-1]
ytr = Ztr[:, -1]
Xte = Zte[:, :-1]
yte = Zte[:, -1]

# train tree ensemble
forest = GradientBoostingClassifier(min_samples_leaf=10)
#forest = RandomForestClassifier(min_samples_leaf=10)
#forest = ExtraTreesClassifier(min_samples_leaf=10)
#forest = AdaBoostClassifier()
forest.fit(Xtr, ytr)

# fit simplified model
Kmax = 10
splitter = DefragModel.parseSLtrees(forest) # parse sklearn tree ensembles into the array of (feature index, threshold)
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
