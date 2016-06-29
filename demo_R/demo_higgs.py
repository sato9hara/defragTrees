# -*- coding: utf-8 -*-
"""
@author: satohara
"""

import sys
sys.path.append('../')

import os
import numpy as np
import pandas as pd
import demo_R

# setting
prefix = 'higgs'
seed = 0
m1 = 10000
m2 = m1 + 10000

# data
df = pd.read_csv('../data/HIGGS.csv', sep=',', header=None)
cols = df.columns.tolist()
cols = cols[1:] + cols[:1]
df = df[cols]

# split
num = len(df)
np.random.seed(seed)
idx = np.random.permutation(num)
df1 = df.ix[idx[:m1], :]
df2 = df.ix[idx[m1:m2], :]

# save
dirname = './result_%s' % (prefix,)
if not os.path.exists(dirname):
    os.mkdir(dirname)
trfile = '%s/%s_train.csv' % (dirname, prefix)
tefile = '%s/%s_test.csv' % (dirname, prefix)
df1.to_csv(trfile, header=None, index=False)
df2.to_csv(tefile, header=None, index=False)

# demo_R
Kmax = 10
restart = 20
M = range(1, 16)
demo_R.run(prefix, Kmax, restart, modeltype='classification', plot=False, treenum=30, maxitr=3000, tol=1)
#demo_R.run(prefix, Kmax, restart, modeltype='classification', plot=False, treenum=30, maxitr=3000, tol=1, M=M, compare=True)
