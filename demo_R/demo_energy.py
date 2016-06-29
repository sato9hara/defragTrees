# -*- coding: utf-8 -*-
"""
@author: Satoshi Hara
"""

import sys
sys.path.append('../')

import os
import numpy as np
import pandas as pd
import demo_R

# setting
prefix = 'energy'
seed = 0
ratio = 0.5

# data
df = pd.read_csv('../data/energy.csv', sep=',', header=None)
df = df.drop([9, 10, 11], 1)

# split
num = len(df)
m = int(np.ceil(ratio * num))
np.random.seed(seed)
idx = np.random.permutation(num)
df1 = df.ix[idx[:m], :]
df2 = df.ix[idx[m:], :]

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
featurename = ('Relative Compactness', 'Surface Area', 'Wall Area', 'Roof Area', 'Overall Height', 'Orientation', 'Glazing Area', 'Glazing Area Distribution')
demo_R.run(prefix, Kmax, restart, featurename=featurename, plot=False)
#demo_R.run(prefix, Kmax, restart, featurename=featurename, plot=False, M=M, compare=True)
