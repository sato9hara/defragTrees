# -*- coding: utf-8 -*-
"""
@author: Satoshi Hara
"""

#import sys
#sys.path.append('../')

import os
import numpy as np
import paper_sub

# setting
prefix = 'synthetic'
seed = 2
num = 1000
dim = 2

# data - train
np.random.seed(seed)
Xtr = np.random.rand(num, dim)
ytr = np.zeros(num)
ytr[(Xtr[:, 0] < 0.5) * (Xtr[:, 1] < 0.5)] = 0
ytr[(Xtr[:, 0] >= 0.5) * (Xtr[:, 1] < 0.5)] = 1
ytr[(Xtr[:, 0] < 0.5) * (Xtr[:, 1] >= 0.5)] = 1
ytr[(Xtr[:, 0] >= 0.5) * (Xtr[:, 1] >= 0.5)] = 0
ytr += 0.1 * np.random.randn(num)

# data - test
Xte = np.random.rand(num, dim)
yte = np.zeros(num)
yte[(Xte[:, 0] < 0.5) * (Xte[:, 1] < 0.5)] = 0
yte[(Xte[:, 0] >= 0.5) * (Xte[:, 1] < 0.5)] = 1
yte[(Xte[:, 0] < 0.5) * (Xte[:, 1] >= 0.5)] = 1
yte[(Xte[:, 0] >= 0.5) * (Xte[:, 1] >= 0.5)] = 0
yte += 0.1 * np.random.randn(num)

# save
dirname = './result_%s' % (prefix,)
if not os.path.exists(dirname):
    os.mkdir(dirname)
trfile = '%s/%s_train.csv' % (dirname, prefix)
tefile = '%s/%s_test.csv' % (dirname, prefix)
np.savetxt(trfile, np.c_[Xtr, ytr], delimiter=',')
np.savetxt(tefile, np.c_[Xte, yte], delimiter=',')

# demo_R
Kmax = 10
restart = 20
M = range(1, 16)
paper_sub.run(prefix, Kmax, restart, plot=True)
#paper_sub.run(prefix, Kmax, restart, plot=True, M=M, compare=True)
