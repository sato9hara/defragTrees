# -*- coding: utf-8 -*-
"""
@author: Satoshi Hara
"""

import sys
sys.path.append('../')
sys.path.append('./baselines/')

import os
import time
import numpy as np
import pandas as pd
from sklearn.externals import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from defragTrees import DefragModel
from Baselines import inTreeModel, NHarvestModel, DTreeModel, BTreeModel
import RulePlotter

def run(prefix, Kmax, restart, M=range(1, 11), featurename=[], modeltype='regression', plot=False, plot_line=[], compare=False, treenum=10, maxitr=100, tol=1e-6):
    
    # data
    dirname = './result/result_%s' % (prefix,)
    trfile = '%s/%s_train.csv' % (dirname, prefix)
    tefile = '%s/%s_test.csv' % (dirname, prefix)
    Ztr = pd.read_csv(trfile, delimiter=',', header=None).values
    Xtr = Ztr[:, :-1]
    ytr = Ztr[:, -1]
    Zte = pd.read_csv(tefile, delimiter=',', header=None).values
    Xte = Zte[:, :-1]
    yte = Zte[:, -1]
    
    # build R random forest
    if modeltype == 'regression':
        os.system('Rscript ./baselines/buildRegForest.R %s %s %s %d 0' % (trfile, tefile, dirname, treenum))
    elif modeltype == 'classification':
        os.system('Rscript ./baselines/buildClfForest.R %s %s %s %d 0' % (trfile, tefile, dirname, treenum))
    splitter = DefragModel.parseRtrees('%s/forest/' % (dirname,))
    
    # Tree Ensemble Interpretation
    print('----- Model Simplification -----')
    mdl = DefragModel(restart=restart, modeltype=modeltype, maxitr=maxitr, tol=tol)
    start = time.time()
    mdl.fit(Xtr, ytr, splitter, Kmax, fittype='FAB', featurename=featurename)
    end = time.time()
    joblib.dump(mdl, '%s/%s_defrag.mdl' % (dirname, prefix), compress=9)
    t = np.c_[len(mdl.rule_), np.array([(end - start) / restart])]
    np.savetxt('%s/%s_defrag_FAB_result.txt' % (dirname, prefix), t, delimiter=',')
    
    # Tree Ensemble Result
    df = pd.read_csv('%s/pred_test.csv' % (dirname,), header=None, delimiter=' ')
    z = df.ix[:, 1].values
    if modeltype == 'regression':
        score = np.mean((yte - z)**2)
    elif modeltype == 'classification':
        score = np.mean(yte != z)
    coll = 1.0
    print()
    print('<< Tree Ensemble >>')
    print('----- Evaluated Results -----')
    print('Test Error = %f' % (score,))
    print('Rule Overlap = %f' % (coll,))
    print()
    
    # defragTrees Result
    score, cover, coll = mdl.evaluate(Xte, yte)
    print()
    print('<< defragTrees >>')
    print('----- Evaluated Results -----')
    print('Test Error = %f' % (score,))
    print('Rule Overlap = %f' % (coll,))
    print()
    if plot:
        RulePlotter.plotRule(mdl, Xte, 0, 1, filename='%s/%s_defrag.pdf' % (dirname, prefix), plot_line=plot_line)
    else:
        print('----- Found Rules -----')
        print(mdl)
    
    # inTrees Result
    mdl2 = inTreeModel(modeltype=modeltype)
    mdl2.fit(Xtr, ytr, '%s/inTrees.txt' % (dirname,), featurename=featurename)
    joblib.dump(mdl2, '%s/%s_inTrees.mdl' % (dirname, prefix), compress=9)
    score, cover, coll = mdl2.evaluate(Xte, yte)
    print()
    print('<< inTrees >>')
    print('----- Evaluated Results -----')
    print('Test Error = %f' % (score,))
    print('Rule Overlap = %f' % (coll,))
    print()
    if plot:
        RulePlotter.plotRule(mdl2, Xte, 0, 1, filename='%s/%s_inTrees.pdf' % (dirname, prefix), plot_line=plot_line)
    else:
        print('----- Found Rules -----')
        print(mdl2)
    
    # NHarvest Result
    mdl3 = NHarvestModel(modeltype=modeltype)
    mdl3.fit(Xtr, ytr, '%s/nodeHarvest.txt' % (dirname,), featurename=featurename)
    joblib.dump(mdl3, '%s/%s_nodeHarvest.mdl' % (dirname, prefix), compress=9)
    score, cover, coll = mdl3.evaluate(Xte, yte)
    df = pd.read_csv('%s/pred_test_nh.csv' % (dirname,), header=None, delimiter=' ')
    z = df.ix[:, 1].values
    if modeltype == 'regression':
        score = np.mean((yte - z)**2)
    elif modeltype == 'classification':
        score = np.mean(yte != z)
    print()
    print('<< nodeHarvest >>')
    print('----- Evaluated Results -----')
    print('Test Error = %f' % (score,))
    print('Rule Overlap = %f' % (coll,))
    print()
    if plot:
        RulePlotter.plotEachRule(mdl3, Xte, 0, 1, filename='%s/%s_nodeHarvest.pdf' % (dirname, prefix), plot_line=plot_line)
    else:
        print('----- Found Rules -----')
        print(mdl3)
    
    # Born Again Tree Result
    mdl4 = BTreeModel(modeltype=modeltype, njobs=1)
    mdl4.fit(Xtr, ytr, '%s/forest/' % (dirname,), featurename=featurename)
    joblib.dump(mdl4, '%s/%s_BATree.mdl' % (dirname, prefix), compress=9)
    score, cover, coll = mdl4.evaluate(Xte, yte)
    print()
    print('<< Born Again Tree >>')
    print('----- Evaluated Results -----')
    print('Test Error = %f' % (score,))
    print('Rule Overlap = %f' % (coll,))
    print()
    if plot:
        RulePlotter.plotRule(mdl4, Xte, 0, 1, filename='%s/%s_BATree.pdf' % (dirname, prefix), plot_line=plot_line)
    else:
        print('----- Found Rules -----')
        print(mdl4)
    
    # Decision Tree Result - depth = 2
    mdl5 = DTreeModel(modeltype=modeltype, max_depth=[2])
    mdl5.fit(Xtr, ytr, featurename=featurename)
    joblib.dump(mdl5, '%s/%s_DTree2.mdl' % (dirname, prefix), compress=9)
    score, cover, coll = mdl5.evaluate(Xte, yte)
    print()
    print('<< Decision Tree (depth 2) >>')
    print('----- Evaluated Results -----')
    print('Test Error = %f' % (score,))
    print('Rule Overlap = %f' % (coll,))
    print()
    if plot:
        RulePlotter.plotRule(mdl5, Xte, 0, 1, filename='%s/%s_DTree2.pdf' % (dirname, prefix), plot_line=plot_line)
    else:
        print('----- Found Rules -----')
        print(mdl5)
    
    # compare defragTrees FAB and EM
    if compare:
        score, cover, coll = mdl.evaluate(Xte, yte)
        s = []
        c = []
        l = []
        t = []
        for m in M:
            mdlm = DefragModel(restart=restart, modeltype=modeltype, maxitr=maxitr, tol=tol)
            start = time.time()
            mdlm.fit(Xtr, ytr, splitter, m, fittype='EM')
            end = time.time()
            joblib.dump(mdlm, '%s/%s_defrag_EM%02d.mdl' % (dirname, prefix, m), compress=9)
            ss, cc, ll = mdlm.evaluate(Xte, yte)
            s.append(ss)
            c.append(cc)
            l.append(ll)
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
        tt = np.c_[M, np.array(t), np.array(s), np.array(c), np.array(l)]
        np.savetxt('%s/%s_defrag_EM_result.txt' % (dirname, prefix), tt, delimiter=',')
        summary2csv(prefix)
        print('[Times]')
        times(prefix, M)

def summary2csv(prefix):
    res1 = pd.read_csv('./result/result_%s/%s_defrag_EM_result.txt' % (prefix, prefix), header=None).values
    res2 = pd.read_csv('./result/result_%s/%s_defrag_FAB_result.txt' % (prefix, prefix), header=None).values
    mdl = joblib.load('./result/result_%s/%s_defrag.mdl' % (prefix, prefix))
    Z = pd.read_csv('./result/result_%s/%s_test.csv' % (prefix, prefix), header=None).values
    error, cover, coll = mdl.evaluate(Z[:, :-1], Z[:, -1])
    res1 = np.c_[res1, np.array(['EM'] * res1.shape[0])]
    res2 = np.array([[res2[0, 0], res2[0, 1], error, cover, coll, 'FAB']])
    res = np.r_[res1, res2]
    if not os.path.exists('./result/csv/'):
        os.mkdir('./result/csv/')
    df = pd.DataFrame(res)
    df.columns = ['K', 'time', 'error', 'coverage', 'overlap', 'label']
    df.to_csv('./result/csv/compare_%s.csv' % (prefix,), index=None)
    
def times(prefix, M):
    t = []
    mdl = joblib.load('./result/result_%s/%s_defrag.mdl' % (prefix, prefix))
    tt = []
    for defragger in mdl.defragger_:
        tt.append(defragger.time_)
    t.append(tt)
    for m in M:
        mdl = joblib.load('./result/result_%s/%s_defrag_EM%02d.mdl' % (prefix, prefix, m))
        tt = []
        for defragger in mdl.defragger_:
            tt.append(defragger.time_)
        t.append(tt)
    t = np.array(t)
    tm1 = np.mean(t[0, :])
    ts1 = np.std(t[0, :])
    tm2 = np.mean(np.sum(t[1:, :], axis=0))
    ts2 = np.std(np.sum(t[1:, :], axis=0))
    print('FAB > %.3f (%.3f)' % (tm1, ts1))
    print('EM > %.3f (%.3f)' % (tm2, ts2))
    