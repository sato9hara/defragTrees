# -*- coding: utf-8 -*-
"""
@author: Satoshi Hara
"""

import sys
import os
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../'))

import numpy as np
import paper_sub_itr

# setting
prefix = 'synthetic2'
seed = 0
num = 1000
dim = 2
trial = 10

# data
b = 0.9

# data
if not os.path.exists('./result/'):
    os.mkdir('./result/')
dirname = './result/result_%s_itr' % (prefix,)
if not os.path.exists(dirname):
    os.mkdir(dirname)
for t in range(trial):
    
    # data - train
    np.random.seed(seed + t)
    Xtr = np.random.rand(num, dim)
    ytr = np.zeros(num)
    ytr = (Xtr[:, 1] > 0.25 + 0.5 / (1 + np.exp(- 20 * (Xtr[:, 0] - 0.5))) + 0.05 * np.cos(Xtr[:, 0] * 2 * np.pi))
    ytr = np.logical_xor(ytr, np.random.rand(num) > b)
    
    # data - test
    Xte = np.random.rand(num, dim)
    yte = np.zeros(num)
    yte = (Xte[:, 1] > 0.25 + 0.5 / (1 + np.exp(- 20 * (Xte[:, 0] - 0.5))) + 0.05 * np.cos(Xte[:, 0] * 2 * np.pi))
    yte = np.logical_xor(yte, np.random.rand(num) > b)
    
    # save
    dirname2 = '%s/result_%02d' % (dirname, t)
    if not os.path.exists(dirname2):
        os.mkdir(dirname2)
    trfile = '%s/%s_train_%02d.csv' % (dirname2, prefix, t)
    tefile = '%s/%s_test_%02d.csv' % (dirname2, prefix, t)
    np.savetxt(trfile, np.c_[Xtr, ytr], delimiter=',')
    np.savetxt(tefile, np.c_[Xte, yte], delimiter=',')

# demo_R
Kmax = 10
restart = 20
njobs = 10
treenum = 100
paper_sub_itr.run(prefix, Kmax, restart, trial, treenum=treenum, modeltype='classification', njobs=njobs)
