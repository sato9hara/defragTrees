# -*- coding: utf-8 -*-
"""
@author: Satoshi Hara
"""

import sys
sys.path.append('../')

import os
import time
import numpy as np
import pandas as pd
from sklearn.externals import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from defragTrees import DefragModel
from inTrees import inTreeModel, DTreeModel

def run(prefix, Kmax, restart, M=range(1, 16), featurename=[], modeltype='regression', plot=False, compare=False, treenum=10, maxitr=100, tol=1e-6):
    
    # data
    dirname = './result_%s' % (prefix,)
    trfile = '%s/%s_train.csv' % (dirname, prefix)
    tefile = '%s/%s_test.csv' % (dirname, prefix)
    Ztr = np.loadtxt(trfile, delimiter=',')
    Xtr = Ztr[:, :-1]
    ytr = Ztr[:, -1]
    Zte = np.loadtxt(tefile, delimiter=',')
    Xte = Zte[:, :-1]
    yte = Zte[:, -1]
    
    # build R random forest
    if modeltype == 'regression':
        os.system('rscript buildRegForest.R %s %s ./result_%s %d 0' % (trfile, tefile, prefix, treenum))
    elif modeltype == 'classification':        
        os.system('rscript buildClfForest.R %s %s ./result_%s %d 0' % (trfile, tefile, prefix, treenum))
    splitter = DefragModel.parseRtrees('%s/forest/' % (dirname,))
    
    # Tree Ensemble Interpretation
    print('----- Model Simplification -----')
    mdl = DefragModel(restart=restart, modeltype=modeltype, maxitr=maxitr, tol=tol)
    start = time.time()
    mdl.fit(ytr, Xtr, splitter, Kmax, fittype='FAB', featurename=featurename)
    end = time.time()
    joblib.dump(mdl, '%s/%s_defrag.mdl' % (dirname, prefix), compress=9)
    t = np.c_[len(mdl.rule_), np.array([(end - start) / restart])]
    np.savetxt('%s/%s_defrag_time.txt' % (dirname, prefix), t, delimiter=',')
    
    # Tree Ensemble Result
    df = pd.read_csv('./result_%s/pred_test.csv' % (prefix,), header=None, delimiter=' ')
    z = df.ix[:, 1].values
    if modeltype == 'regression':
        score = np.mean((yte - z)**2)
    elif modeltype == 'classification':
        score = np.mean(yte != z)
    cover = 1.0
    print()
    print('<< Tree Ensemble >>')
    print('----- Evaluated Results -----')
    print('Test Error = %f' % (score,))
    print('Test Coverage = %f' % (cover,))
    print()
    
    # defragTrees Result
    score, cover = mdl.evaluate(yte, Xte)
    print()
    print('<< defragTrees >>')
    print('----- Evaluated Results -----')
    print('Test Error = %f' % (score,))
    print('Test Coverage = %f' % (cover,))
    print()
    if plot:
        mdl.plotRule(Xte, 0, 1, filename='%s/%s_defrag.pdf' % (dirname, prefix))
    else:
        print('----- Found Rules -----')
        print(mdl)
    
    # inTrees Result
    mdl2 = inTreeModel(modeltype=modeltype)
    mdl2.fit(ytr, Xtr, './result_%s/inTrees.txt' % (prefix,), featurename=featurename)
    joblib.dump(mdl2, '%s/%s_inTrees.mdl' % (dirname, prefix), compress=9)
    score, cover = mdl2.evaluate(yte, Xte, rnum=10)
    print()
    print('<< inTrees >>')
    print('----- Evaluated Results -----')
    print('Test Error = %f' % (score,))
    print('Test Coverage = %f' % (cover,))
    print()
    if plot:
        mdl2.plotRule(Xte, 0, 1, filename='%s/%s_inTrees.pdf' % (dirname, prefix), rnum=10)
    else:
        print('----- Found Rules -----')
        print(mdl2)
    
    # Decision Tree Result
    mdl3 = DTreeModel(modeltype=modeltype)
    mdl3.fit(ytr, Xtr, featurename=featurename)
    joblib.dump(mdl, '%s/%s_DTree.mdl' % (dirname, prefix), compress=9)
    score, cover = mdl3.evaluate(yte, Xte)
    print()
    print('<< Decision Tree >>')
    print('----- Evaluated Results -----')
    print('Test Error = %f' % (score,))
    print('Test Coverage = %f' % (cover,))
    print()
    if plot:
        mdl3.plotRule(Xte, 0, 1, filename='%s/%s_DTree.pdf' % (dirname, prefix))
    else:
        print('----- Found Rules -----')
        print(mdl3)
    
    # compare defragTrees FAB and EM
    if compare:
        score, cover = mdl.evaluate(yte, Xte)
        s = []
        t = []
        for m in M:
            mdlm = DefragModel(restart=restart, modeltype=modeltype, maxitr=maxitr, tol=tol)
            start = time.time()
            mdlm.fit(ytr, Xtr, splitter, m, fittype='EM')
            end = time.time()
            joblib.dump(mdlm, '%s/%s_defrag_EM%02d.mdl' % (dirname, prefix, m), compress=9)
            ss, cc = mdlm.evaluate(yte, Xte)
            s.append(ss)
            t.append((end - start) / restart)
        plt.plot(M, s, 'bo-', label='EM w/ fixed K', markersize=12)
        plt.plot(len(mdl.rule_), score, 'r*', markersize=28, label='FAB')
        plt.xlabel('# of Components K', fontsize=22)
        plt.ylabel('Mean Square Error', fontsize=22)
        plt.legend(loc = 'upper right', numpoints=1, fontsize=24)
        plt.xlim([min(M), max(M)])
        plt.show()
        plt.savefig('%s/%s_compare.pdf' % (dirname, prefix), format="pdf", bbox_inches="tight")
        plt.close()
        t = np.c_[M, np.array(t)]
        np.savetxt('%s/%s_defrag_EM_time.txt' % (dirname, prefix), t, delimiter=',')
        