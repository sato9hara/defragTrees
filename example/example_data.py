# -*- coding: utf-8 -*-
"""
@author: Satoshi Hara
"""

import numpy as np

# setting
seed = 0
num = 1000
dim = 2

# data - train
np.random.seed(seed)
Xtr = np.random.rand(num, dim)
ytr = np.zeros(num)
ytr = np.logical_xor(Xtr[:, 0] > 0.5, Xtr[:, 1] > 0.5)
ytr = np.logical_xor(ytr, np.random.rand(num) > 0.9)

# data - test
Xte = np.random.rand(num, dim)
yte = np.zeros(num)
yte = np.logical_xor(Xte[:, 0] > 0.5, Xte[:, 1] > 0.5)
yte = np.logical_xor(yte, np.random.rand(num) > 0.9)

# save
np.savetxt('./train.csv', np.c_[Xtr, ytr], delimiter=',')
np.savetxt('./test.csv', np.c_[Xte, yte], delimiter=',')
