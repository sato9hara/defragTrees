# -*- coding: utf-8 -*-
"""
@author: Satoshi Hara
"""

import sys
import os
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('../'))

import numpy as np
import pandas as pd
import paper_sub_itr

 setting
prefix = 'higgs'
seed = 0
m1 = 5000
m2 = m1 + 5000
trial = 10

# data
df = pd.read_csv('./data/HIGGS.csv', sep=',', header=None)
cols = df.columns.tolist()
cols = cols[1:] + cols[:1]
df = df[cols]
num = len(df)

# data
if not os.path.exists('./result/'):
    os.mkdir('./result/')
dirname = './result/result_%s_itr' % (prefix,)
if not os.path.exists(dirname):
    os.mkdir(dirname)
for t in range(trial):
    
    # data - train & test
    np.random.seed(seed + t)
    idx = np.random.permutation(num)
    df1 = df.ix[idx[:m1], :]
    df2 = df.ix[idx[m1:m2], :]
    
    # save
    dirname2 = '%s/result_%02d' % (dirname, t)
    if not os.path.exists(dirname2):
        os.mkdir(dirname2)
    trfile = '%s/%s_train_%02d.csv' % (dirname2, prefix, t)
    tefile = '%s/%s_test_%02d.csv' % (dirname2, prefix, t)
    df1.to_csv(trfile, header=None, index=False)
    df2.to_csv(tefile, header=None, index=False)

# demo_R
Kmax = 10
restart = 20
njobs = 10
maxitr = 3000
tol = 1e-2
treenum = 100
paper_sub_itr.run(prefix, Kmax, restart, trial, treenum=treenum, modeltype='classification', maxitr=maxitr, tol=tol, njobs=njobs)
