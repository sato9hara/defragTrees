# -*- coding: utf-8 -*-
"""
@author: Satoshi Hara
"""

import numpy as np
from collections import Counter
from multiprocessing import Pool

def argwrapper(args):
    return args[0](*args[1:])

def fitBATreeCV(X, y, mdl, modeltype='regression', max_depth=[2, 3, 4], min_samples_split=[10, 20, 30], cv=5, seed=0, smear_num=100, njobs=1):
    np.random.seed(seed)
    idx = np.mod(np.random.permutation(X.shape[0]), cv)
    E = np.zeros((len(max_depth), len(min_samples_split), cv))
    s = 0
    if njobs == 1:
        for k in range(cv):
            Xtr = X[idx != k, :]
            ytr = y[idx != k]
            Xte = X[idx == k, :]
            yte = y[idx == k]
            for i, depth in enumerate(max_depth):
                for j, split in enumerate(min_samples_split):
                    i, j, k, err = subfit(i, j, k, Xtr, ytr, Xte, yte, mdl, modeltype, depth, split, smear_num, s)
                    E[i, j, k] = err
                    s += 1
    else:
        pool = Pool(processes = njobs)
        args = []
        for k in range(cv):
            Xtr = X[idx != k, :]
            ytr = y[idx != k]
            Xte = X[idx == k, :]
            yte = y[idx == k]
            for i, depth in enumerate(max_depth):
                for j, split in enumerate(min_samples_split):
                    args.append((subfit, i, j, k, Xtr, ytr, Xte, yte, mdl, modeltype, depth, split, smear_num, s))
                    s += 1
        res = pool.map(argwrapper, args)
        pool.close()
        pool.join()
        for r in res:
            E[r[0], r[1], r[2]] = r[3]
    E = np.mean(E, axis=2)
    i = np.sum(E == np.min(E), axis=1).nonzero()[0][0]
    j = np.sum(E == np.min(E), axis=0).nonzero()[0][0]
    tree = BATreeModel(modeltype=modeltype, max_depth=max_depth[i], min_samples_split=min_samples_split[j], smear_num=smear_num, seed=seed)
    tree.fit(X, y, mdl)
    return tree

def subfit(i, j, k, Xtr, ytr, Xte, yte, mdl, modeltype, max_depth, min_samples_split, smear_num, seed):
    tree = BATreeModel(modeltype=modeltype, max_depth=max_depth, min_samples_split=min_samples_split, smear_num=smear_num, seed=seed)
    tree.fit(Xtr, ytr, mdl)
    zte = tree.predict(Xte)
    if modeltype == 'regression':
        err = np.mean((yte - zte)**2)
    elif modeltype == 'classification':
        err = np.mean(yte != zte)
    return i, j, k, err

class BATreeModel(object):
    def __init__(self, modeltype='regression', max_depth=3, min_samples_split=10, search_num=100, smear_num=100, palt=0.5, seed=0):
        self.modeltype_ = modeltype
        self.max_depth_ = max_depth
        self.min_samples_split_ = min_samples_split
        self.search_num_ = search_num
        self.smear_num_ = smear_num
        self.palt_ = palt
        self.seed_ = 0
        self.left_ = []
        self.right_ = []
        self.index_ = []
        self.threshold_ = []
        self.pred_ = []
    
    def predict(self, X):
        y = np.zeros(X.shape[0])
        for n in range(X.shape[0]):
            i = len(self.pred_) - 1
            while True:
                if self.left_[i] == -1:
                    if self.modeltype_ == 'regression':
                        y[n] = self.pred_[i]
                    elif self.modeltype_ == 'classification':
                        y[n] = np.argmax(self.pred_[i])
                    break
                else:
                    if X[n, self.index_[i]] <= self.threshold_[i]:
                        i = self.left_[i]
                    else:
                        i = self.right_[i]
        if self.modeltype_ == 'classification':
            y = y.astype(np.int64)
        return y
    
    def fit(self, X, y, mdl):
        self.dim_ = X.shape[1]
        if self.modeltype_ == 'classification':
            self.C_ = np.unique(y).size
        num = X.shape[0]
        idx = np.arange(num)
        box = np.zeros((2, self.dim_))
        for d in range(self.dim_):
            box[0, d] = np.min(X[:, d])
            box[1, d] = np.max(X[:, d])
        self.__split(0, X, y, idx, mdl, self.left_, self.right_, self.index_, self.threshold_, self.pred_, box)
        
    def __split(self, depth, X, y, idx, mdl, left, right, index, threshold, pred, box):
        if self.modeltype_ == 'regression':
            z = np.mean(y[idx])
        elif self.modeltype_ == 'classification':
            key = list(Counter(y[idx]).keys())
            count = np.array(list(Counter(y[idx]).values()))
            z = np.array([(count[key.index(c)] / np.sum(count) if c in key else 0) for c in range(self.C_)])
        if depth >= self.max_depth_ or idx.size <= self.min_samples_split_:
            left.append(-1)
            right.append(-1)
            index.append(-1)
            threshold.append(0)
            pred.append(z)
            return len(pred) - 1
        XX, yy = self.__smear(X, mdl, box, self.smear_num_, palt=self.palt_, seed=self.seed_+len(pred))
        f_opt = np.inf
        index_opt = -1
        threshold_opt = 0
        np.random.seed(self.seed_+len(pred))
        for d in range(self.dim_):
            t = np.unique(X[idx, d])[:-1] + 0.5 * np.diff(np.unique(X[idx, d]))
            if t.size <= 1:
                continue
            if t.size > self.search_num_:
                t = t[np.random.permutation(t.size)[:self.search_num_]]
            for tt in t:
                yl = yy[XX[:, d] <= tt]
                yr = yy[XX[:, d] > tt]
                if self.modeltype_ == 'regression':
                    f = np.sum((yl - np.mean(yl))**2) + np.sum((yr - np.mean(yr))**2)
                elif self.modeltype_ == 'classification':
                    key = list(Counter(yl).keys())
                    count = np.array(list(Counter(yl).values()))
                    pl = np.array([(count[key.index(c)] / np.sum(count) if c in key else 0) for c in range(self.C_)])
                    key = list(Counter(yr).keys())
                    count = np.array(list(Counter(yr).values()))
                    pr = np.array([(count[key.index(c)] / np.sum(count) if c in key else 0) for c in range(self.C_)])
                    f = yl.size * pl.dot(1 - pl) + yr.size * pr.dot(1 - pr)
                if f < f_opt:
                    f_opt = f
                    index_opt = d
                    threshold_opt = tt
        idxl = idx[X[idx, index_opt] <= threshold_opt]
        idxr = idx[X[idx, index_opt] > threshold_opt]
        if idxl.size == 0 or idxr.size == 0:
            left.append(-1)
            right.append(-1)
            index.append(-1)
            threshold.append(0)
            pred.append(z)
            return len(pred) - 1
        boxl = box.copy()
        boxl[1, index_opt] = threshold_opt
        boxr = box.copy()
        boxr[0, index_opt] = threshold_opt
        l = self.__split(depth+1, X, y, idxl, mdl, left, right, index, threshold, pred, boxl)
        r = self.__split(depth+1, X, y, idxr, mdl, left, right, index, threshold, pred, boxr)
        left.append(l)
        right.append(r)
        index.append(index_opt)
        threshold.append(threshold_opt)
        pred.append(z)
        return len(pred) - 1
    
    def __smear(self, X, mdl, box, num, palt=0.5, seed=0):
        dim = X.shape[1]
        idx = []
        for d in range(dim):
            idx.append(((X[:, d] >= box[0, d]) * (X[:, d] <= box[1, d])).nonzero()[0])
        XX = np.zeros((num, dim))
        np.random.seed(seed)
        n = 0
        while True:
            m = np.random.randint(X.shape[0])
            flg = True
            for d in range(dim):
                if np.random.rand() > palt:
                    XX[n, d] = X[m, d]
                else:
                    XX[n, d] = X[idx[d][np.random.randint(idx[d].size)], d]
                flg *= (XX[n, d] >= box[0, d])
                flg *= (XX[n, d] <= box[1, d])
                if not flg:
                    break
            if flg:
                n += 1
            if n >= num:
                break
        yy = mdl.predict(XX)
        return XX, yy
        