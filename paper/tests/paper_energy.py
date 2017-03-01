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
prefix = 'energy'
seed = 1
ratio = 0.5

# data
df = pd.read_csv('./data/energy.csv', sep=',', header=None)
df = df.drop([9, 10, 11], 1)

# split
num = len(df)
m = int(np.ceil(ratio * num))
np.random.seed(seed)
idx = np.random.permutation(num)
df1 = df.ix[idx[:m], :]
df2 = df.ix[idx[m:], :]

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
restart = 200
treenum = 100
M = range(1, 11)
featurename = ('Relative Compactness', 'Surface Area', 'Wall Area', 'Roof Area', 'Overall Height', 'Orientation', 'Glazing Area', 'Glazing Area Distribution')
#paper_sub.run(prefix, Kmax, restart, modeltype='regression', plot=False, treenum=treenum, featurename=featurename)
paper_sub.run(prefix, Kmax, restart, treenum=treenum, modeltype='regression', featurename=featurename, plot=False, M=M, compare=True)
