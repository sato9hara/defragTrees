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
prefix = 'spambase'
seed = 0
m1 = 1000
m2 = m1 + 1000

# data
df = pd.read_csv('./data/spambase.data', header=None)

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
M = range(1, 11)
#paper_sub.run(prefix, Kmax, restart, modeltype='classification', plot=False, treenum=treenum)
paper_sub.run(prefix, Kmax, restart, treenum=treenum, modeltype='classification', plot=False, M=M, compare=True)
