# -*- coding: utf-8 -*-
"""
@author: Satoshi Hara
"""

import numpy as np

# setting
seed = 1
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
np.savetxt('./train.csv', np.c_[Xtr, ytr], delimiter=',')
np.savetxt('./test.csv', np.c_[Xte, yte], delimiter=',')
