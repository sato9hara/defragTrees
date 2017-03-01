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
import paper_sub

# setting
prefix = 'miniboone'
seed = 0
m1 = 5000
m2 = m1 + 5000

# data
df = pd.read_csv('./data/MiniBooNE_PID.txt', header=None, delim_whitespace=True, skiprows=1)
num = df.shape[0]
idx = 36499
y = np.r_[np.ones(idx), np.zeros(num-idx)]
df[df.columns[-1]+1] = pd.Series(y)
idx = np.where(np.min(df, axis=1) < -900)[0]
df = df.drop(idx)
df = pd.DataFrame(df.values)

# split
num = len(df)
np.random.seed(seed)
idx = np.random.permutation(num)
df1 = df.ix[idx[:m1], :]
df2 = df.ix[idx[m1:m2], :]

# save
if not os.path.exists('./result/'):
    os.mkdir('./result/')
dirname = './result/result_%s' % (prefix,)
if not os.path.exists(dirname):
    os.mkdir(dirname)
trfile = '%s/%s_train.csv' % (dirname, prefix)
tefile = '%s/%s_test.csv' % (dirname, prefix)
df1.to_csv(trfile, header=None, index=False)
df2.to_csv(tefile, header=None, index=False)

# demo_R
Kmax = 10
restart = 20
treenum = 100
maxitr = 3000
tol = 1e-2
M = range(1, 11)
#paper_sub.run(prefix, Kmax, restart, modeltype='classification', plot=False, treenum=treenum, maxitr=maxitr, tol=tol)
paper_sub.run(prefix, Kmax, restart, treenum=treenum, modeltype='classification', maxitr=maxitr, tol=tol, plot=False, M=M, compare=True)
