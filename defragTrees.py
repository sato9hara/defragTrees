# -*- coding: utf-8 -*-
"""
@author: satohara
"""

import numpy as np
import pandas as pd
import glob
import re
from sklearn import tree
from sklearn.grid_search import GridSearchCV

import pylab as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colorbar as colorbar

#************************
# Default Class
#************************
class RuleModel(object):
    def __init__(self, modeltype='regression'):
        self.modeltype_ = modeltype
        self.dim_ = 0
        self.featurename_ = []
        self.rule_ = []
        self.pred_ = []
        self.pred_default_ = []
    
    def __str__(self):
        s = ''
        for i in range(len(self.rule_)):
            s += '[Rule %2d]\n' % (i+1,)
            if self.modeltype_ == 'regression':
                s += 'z = %f when\n' % (self.pred_[i],)
            elif self.modeltype_ == 'classification':
                s += 'z = %d when\n' % (self.pred_[i],)
            box, vmin, vmax = self.__r2box(self.rule_[i], self.dim_)
            for d in range(box.shape[1]):
                if box[0, d] == vmin[d] and box[1, d] == vmax[d]:
                    pass
                elif box[0, d] > vmin[d] and box[1, d] < vmax[d]:
                    s += '\t %f <= %s < %f\n' % (box[0, d], self.featurename_[d], box[1, d])
                elif box[0, d] == vmin[d]:
                    s += '\t %s < %f\n' % (self.featurename_[d], box[1, d])
                elif box[1, d] == vmax[d]:
                    s += '\t %s >= %f\n' % (self.featurename_[d], box[0, d])
            s += '\n'
        s += '[Otherwise]\n'
        if self.modeltype_ == 'regression':
            s += 'z = %f\n' % (self.pred_default_,)
        elif self.modeltype_ == 'classification':
            s += 'z = %d\n' % (self.pred_default_,)
        return s
    
    #************************
    # Common Methods
    #************************
    def predict(self, X, rnum=-1):
        num = X.shape[0]
        if rnum <= 0:
            rnum = len(self.rule_)
        else:
            rnum = min(len(self.rule_), rnum)
        y = np.zeros((num, rnum+1))
        y[:, -1] = np.nan
        for i in range(rnum):
            r = self.rule_[i]
            flg = np.ones(num)
            for l in r:
                if l[1] == 0:
                    flg *= (X[:, l[0]-1] <= l[2])
                else:
                    flg *= (X[:, l[0]-1] >= l[2])
            y[flg < 0.5, i] = np.nan
            y[flg > 0.5, i] = self.pred_[i]
        y[np.sum(np.isnan(y[:, :-1]), axis=1) == rnum, -1] = self.pred_default_
        return y
        
    def evaluate(self, y, X, rnum=-1):
        z = self.predict(X, rnum=rnum)
        num = X.shape[0]
        err = 0
        cov = 0
        count = 0
        for n in range(num):
            if np.isnan(z[n, -1]):
                cov += 1
            count += 1
            v = z[n, ~np.isnan(z[n, :])]
            tmp = 0
            if self.modeltype_ == 'regression':
                tmp += np.mean((v - y[n])**2)
            elif self.modeltype_ == 'classification':
                tmp += np.mean(v != y[n])
            err += tmp / v.size
        return err / count, cov / num
    
    def __r2box(self, r, dim):
        vmin = np.array([-np.inf] * dim)
        vmax = np.array([np.inf] * dim)
        box = np.c_[vmin, vmax].T
        for rr in r:
            if rr[1] == 0:
                box[1, rr[0]-1] = np.minimum(box[1, rr[0]-1], rr[2])
            else:
                box[0, rr[0]-1] = np.maximum(box[0, rr[0]-1], rr[2])
        return box, vmin, vmax
    
    def setfeaturename(self, featurename):
        if len(featurename) > 0:
            self.featurename_ = featurename
        else:
            self.featurename_ = []
            for d in range(self.dim_):
                self.featurename_.append('x_%d' % (d+1,))
    
    def setdefaultpred(self, y):
        if self.modeltype_ == 'regression':
            self.pred_default_ = np.mean(y)
        elif self.modeltype_ == 'classification':
            w = np.unique(y)
            for i in range(w.size):
                w[i] = np.sum(y == w[i])
            w = np.argmax(w)
            self.pred_default_ = w
    
    def plotRule(self, X, d1, d2, alpha=0.8, filename='', rnum=-1):
        cmap = cm.get_cmap('cool')
        fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw = {'width_ratios':[19, 1]})
        if rnum <= 0:
            rnum = len(self.rule_)
        else:
            rnum = min(len(self.rule_), rnum)
        for i in range(rnum):
            r = self.rule_[i]
            box, vmin, vmax = self.__r2boxWithX(r, X)
            if self.modeltype_ == 'regression':
                c = cmap(self.pred_[i])
            elif self.modeltype_ == 'classification':
                r = self.pred_[i] / (np.unique(self.pred_).size - 1)
                c = cmap(r)
            ax1.add_patch(pl.Rectangle(xy=[box[0, d1], box[0, d2]], width=(box[1, d1] - box[0, d1]), height=(box[1, d2] - box[0, d2]), facecolor=c, linewidth='2.0', alpha=alpha))
        ax1.set_xlabel('x1', size=22)
        ax1.set_ylabel('x2', size=22)
        ax1.set_title('Simplified Model (K = %d)' % (rnum,), size=28)
        colorbar.ColorbarBase(ax2, cmap=cmap, format='%.1f')
        ax2.set_ylabel('Predictor z', size=22)
        plt.show()
        if not filename == '':
            plt.savefig(filename, format="pdf", bbox_inches="tight")
            plt.close()
    
    def __r2boxWithX(self, r, X):
        vmin = np.min(X, axis=0)
        vmax = np.max(X, axis=0)
        box = np.c_[vmin, vmax].T
        for rr in r:
            if rr[1] == 0:
                box[1, rr[0]-1] = np.minimum(box[1, rr[0]-1], rr[2])
            else:
                box[0, rr[0]-1] = np.maximum(box[0, rr[0]-1], rr[2])
        return box, vmin, vmax
    
    
#************************
# Defrag Class
#************************
class DefragModel(RuleModel):
    def __init__(self, modeltype='regression', maxitr=100, tol=1e-6, eps=1e-10, delta=1e-8, kappa=1e-3, seed=0, restart=10, verbose=0):
        super().__init__(modeltype=modeltype)
        self.maxitr_ = maxitr
        self.tol_ = tol
        self.eps_ = eps
        self.delta_ = delta
        self.kappa_ = kappa
        self.seed_ = seed
        self.restart_ = restart
        self.verbose_ = verbose
    
    #************************
    # Fit and Related Methods
    #************************
    def fit(self, y, X, splitter, K, fittype='EM', featurename=[]):
        self.dim_ = X.shape[1]
        self.setfeaturename(featurename)
        self.setdefaultpred(y)
        opt_score = np.inf
        opt_itr = 0
        for itr in range(self.restart_):
            if fittype == 'EM':
                h, E, A, Q, f = self.__fitEM(y, X, splitter, K, self.seed_+itr)
            if fittype == 'FAB':
                h, E, A, Q, G, f = self.__fitFAB(y, X, splitter, K, self.seed_+itr)
                h = h[:, G]
                E = E[:, G]
            rule, pred = self.__param2rules(y, X, splitter, h, E, kappa=self.kappa_, modeltype=self.modeltype_)
            if itr == 0:
                self.rule_ = rule
                self.pred_ = pred
                score, coverage = self.evaluate(y, X)
                opt_score = score
                opt_itr = 0
            else:
                tmp_rule = self.rule_
                tmp_pred = self.pred_
                self.rule_ = rule
                self.pred_ = pred
                score, coverage = self.evaluate(y, X)
                if score > opt_score:
                    self.rule_ = tmp_rule
                    self.pred_ = tmp_pred
                else:
                    opt_score = score
                    opt_itr = itr
            print('[Model %3d] TrainingError = %.2f' % (itr, score))
        print('Optimal Model >> Model %3d, TrainingError = %.2f' % (opt_itr, opt_score))
        
    def __param2rules(self, y, X, splitter, h, E, kappa=1e-3, modeltype='regression'):
        rule = []
        pred = []
        d = np.unique(splitter[:, 0])
        vmin = np.zeros(d.size)
        vmax = np.zeros(d.size)
        for i, dd in enumerate(d):
            vmin[i] = np.min(splitter[splitter[:, 0] == dd, 1]) - kappa
            vmax[i] = np.max(splitter[splitter[:, 0] == dd, 1]) + kappa
        for k in range(h.shape[1]):
            box = np.c_[vmin, vmax].T
            for j in range(E.shape[0]):
                idx = (d == splitter[j, 0])
                if E[j, k] > 1 - kappa:
                    box[0, idx] = np.maximum(box[0, idx], splitter[j, 1])
                elif E[j, k] < kappa:
                    box[1, idx] = np.minimum(box[1, idx], splitter[j, 1])
            subrule = []
            for i, dd in enumerate(d):
                if not box[0, i] == vmin[i]:
                    subrule.append((dd+1, 1, box[0, i]))
                if not box[1, i] == vmax[i]:
                    subrule.append((dd+1, 0, box[1, i]))
            rule.append(subrule)
            if modeltype == 'regression':
                pred.append(h[0, k])
            elif modeltype == 'classification':
                pred.append(np.argmax(h[:, k]))
        idx = np.argsort(np.array(pred))
        rule = np.array(rule)[idx].tolist()
        rule = self.__pruneRule(X, rule)
        pred = np.array(pred)[idx].tolist()
        return rule, pred
    
    def __pruneRule(self, X, rule):
        for i, r in enumerate(rule):
            f = self.__getRuleN(r, X)
            while True:
                flg = True
                for j in range(len(r)):
                    g = self.__getRuleN(r, X, j=j)
                    if g == f:
                        del(r[j])
                        flg = False
                        break
                if flg:
                    break
            rule[i] = r
        return rule
        
    def __getRuleN(self, r, X, j=-1):
        num = X.shape[0]
        flg = np.ones(num)
        for i, l in enumerate(r):
            if i == j:
                continue
            if l[1] == 0:
                flg *= (X[:, l[0]-1] <= l[2])
            else:
                flg *= (X[:, l[0]-1] >= l[2])
        return np.sum(flg)
        
    def __getBinary(self, X, splitter):
        r = splitter.shape[0]
        num = X.shape[0]
        R = np.zeros((num, r))
        for i in range(r):
            R[:, i] = X[:, splitter[i, 0]] >= splitter[i, 1]
        return R
        
    def __fitEM(self, y, X, splitter, K, seed):
        R= self.__getBinary(X, splitter)
        dim = R.shape[1]
        num = X.shape[0]
        
        # initialization of Q, h, E, A
        np.random.seed(seed)
        Q = np.random.rand(num, K)
        Q /= np.sum(Q, axis=1)[:, np.newaxis]
        if self.modeltype_ == 'regression':
            h = np.c_[np.random.randn(K), np.random.rand(K)].T
        elif self.modeltype_ == 'classification':
            C = np.unique(y).size
            h = np.random.rand(C, K)
            h /= np.sum(h, axis=0)[np.newaxis, :]
        E = np.random.rand(dim, K)
        A = np.random.rand(K)
        A /= np.sum(A)
        
        # train
        Qnew = Q.copy()
        hnew = h.copy()
        Enew = E.copy()
        Anew = A.copy()
        f = self.__objEM(y, R, Q, h, E, A, eps=self.eps_, modeltype=self.modeltype_)
        for itr in range(self.maxitr_):
            Qnew = self.__updateQEM(y, R, Q, h, E, A, eps=self.eps_, modeltype=self.modeltype_)
            hnew = self.__updateH(y, Qnew, h, modeltype=self.modeltype_)
            Enew = self.__updateE(R, Qnew, E)
            Anew = self.__updateA(Qnew, A)
            
            # update
            fnew = self.__objEM(y, R, Qnew, hnew, Enew, Anew, eps=self.eps_, modeltype=self.modeltype_)
            if self.verbose_ > 0:
                if np.mod(itr, self.verbose_) == 0:
                    print(itr, fnew, fnew - f)
            if fnew - f < self.tol_:
                break
            Q = Qnew.copy()
            h = hnew.copy()
            E = Enew.copy()
            A = Anew.copy()
            f = fnew
        return h, E, A, Q, f
        
    def __fitFAB(self, y, X, splitter, Kmax, seed):
        R= self.__getBinary(X, splitter)
        dim = R.shape[1]
        num = X.shape[0]
        
        # initialization of Q, h, E, A, K
        np.random.seed(seed)
        Q = np.random.rand(num, Kmax)
        Q /= np.sum(Q, axis=1)[:, np.newaxis]
        if self.modeltype_ == 'regression':
            h = np.c_[np.random.randn(Kmax), np.random.rand(Kmax)].T
        elif self.modeltype_ == 'classification':
            C = np.unique(y).size
            h = np.random.rand(C, Kmax)
            h /= np.sum(h, axis=0)[np.newaxis, :]
        E = np.random.rand(dim, Kmax)
        A = np.random.rand(Kmax)
        A /= np.sum(A)
        K = np.arange(Kmax)
            
        # train
        Qnew = Q.copy()
        hnew = h.copy()
        Enew = E.copy()
        Anew = A.copy()
        Knew = K.copy()
        f = self.__objFAB(y, R, Q, h, E, A, K, eps=self.eps_, modeltype=self.modeltype_)
        for itr in range(self.maxitr_):
            Qnew = Q.copy()
            Qnew[:, K] = self.__updateQFAB(y, R, Q[:, K], h[:, K], E[:, K], A[K], eps=self.eps_, modeltype=self.modeltype_)
            Knew = np.where(np.mean(Qnew, axis=0) > self.delta_)[0]
            hnew = h.copy()
            hnew[:, Knew] = self.__updateH(y, Qnew[:, Knew], h[:, Knew], modeltype=self.modeltype_)
            Enew = E.copy()
            Enew[:, Knew] = self.__updateE(R, Qnew[:, Knew], E[:, Knew])
            Anew = A.copy()
            Anew[Knew] = self.__updateA(Qnew[:, Knew], A[Knew])
            
            # update
            fnew = self.__objFAB(y, R, Qnew, hnew, Enew, Anew, Knew, eps=self.eps_, modeltype=self.modeltype_)
            if self.verbose_ > 0:
                if np.mod(itr, self.verbose_) == 0:
                    print(itr, fnew, fnew - f, Knew)
            if fnew - f < self.tol_:
                break
            Q = Qnew.copy()
            h = hnew.copy()
            E = Enew.copy()
            A = Anew.copy()
            K = Knew.copy()
            f = fnew
        return h, E, A, Q, K, f
        
    def __getLogP(self, k, y, R, h, E, A, eps=1e-10, modeltype='regression'):
        num = y.size
        logP = np.zeros(num)
        if modeltype == 'regression':
            logP = - h[1, k] * ((y - h[0, k])**2) / 2 - np.log(2 * np.pi / h[1, k]) / 2
            t = 2
        elif modeltype == 'classification':
            C = h.shape[0]
            for c in range(C):
                logP[y == c] = np.log(h[c, k])
            t = C
        logP += R.dot(np.log(np.maximum(eps, E[:, k]))) + (1 - R).dot(np.log(np.maximum(eps, 1 - E[:, k])))
        logP += np.log(A[k])
        return logP, t
        
    def __objEM(self, y, R, Q, h, E, A, eps=1e-10, modeltype='regression'):
        K = Q.shape[1]
        f = 0
        for k in range(K):
            logP, t = self.__getLogP(k, y, R, h, E, A, eps=eps, modeltype=modeltype)
            f += Q[:, k].dot(logP)
            f -= Q[:, k].dot(np.log(Q[:, k]))
        return f
    
    def __objFAB(self, y, R, Q, h, E, A, K, eps=1e-10, modeltype='regression'):
        L = R.shape[1]
        f = 0
        for k in K:
            logP, t = self.__getLogP(k, y, R, h, E, A, eps=eps, modeltype=modeltype)
            f += Q[:, k].dot(logP)
            f -= Q[:, k].dot(np.log(Q[:, k]))
            f -= 0.5 * (1 + t + L) * np.log(1 + np.sum(Q[:, k]))
        return f
    
    def __updateQEM(self, y, R, Q, h, E, A, maxitr=1000, tol=1e-6, eps=1e-10, modeltype='regression'):
        K = Q.shape[1]
        F = Q.copy()
        for k in range(K):
            logP, t = self.__getLogP(k, y, R, h, E, A, eps=eps, modeltype=modeltype)
            F[:, k] = logP
        return self.__normExp(F, eps=eps)
    
    def __updateQFAB(self, y, R, Q, h, E, A, maxitr=1000, tol=1e-6, eps=1e-10, modeltype='regression'):
        K = Q.shape[1]
        L = R.shape[1]
        F = Q.copy()
        for k in range(K):
            logP, t = self.__getLogP(k, y, R, h, E, A, eps=eps, modeltype=modeltype)
            F[:, k] = logP
        g = self.__objQ(F, Q, t, L)
        Qnew = Q.copy()
        for itr in range(maxitr):
            S = F - 0.5 * (1 + t + L) / (1 + np.sum(Q, axis=0)[np.newaxis, :])
            Qnew = self.__normExp(S, eps=eps)
            gnew = self.__objQ(F, Qnew, t, L)
            if gnew > g:
                Q = Qnew.copy()
            if gnew - g < tol:
                break
            g = gnew
        return Q
    
    def __objQ(self, F, Q, t, L):
        f = 0
        for k in range(Q.shape[1]):
            f += Q[:, k].dot(F[:, k])
            f -= Q[:, k].dot(np.log(Q[:, k]))
            f -= 0.5 * (1 + t + L) * np.log(1 + np.sum(Q[:, k]))
        return f
        
    def __updateH(self, y, Q, h, modeltype='regression'):
        K = Q.shape[1]
        for k in range(K):
            if modeltype == 'regression':
                h[0, k] = Q[:, k].dot(y) / np.sum(Q[:, k])
                h[1, k] = np.sum(Q[:, k]) / Q[:, k].dot((y - h[0, k])**2)
            elif modeltype == 'classification':
                C = h.shape[0]
                for c in range(C):
                    h[c, k] = np.sum(Q[y == c, k]) / np.sum(Q[:, k])
        return h
    
    def __updateE(self, R, Q, E):
        K = Q.shape[1]
        for k in range(K):
            E[:, k] = R.T.dot(Q[:, k]) / np.sum(Q[:, k])
        return E
    
    def __updateA(self, Q, A):
        return np.sum(Q, axis=0) / np.sum(Q)
    
    def __normExp(self, A, eps=1e-10):
        if A.shape[1] == 1:
            return np.ones((A.shape[0], 1))
        else:
            A -= np.max(A, axis=1)[:, np.newaxis]
            A = np.exp(A)
            A /= np.sum(A, axis=1)[:, np.newaxis]
            A += eps
            A /= np.sum(A, axis=1)[:, np.newaxis]
            return A
        
    #************************
    # Static Methods
    #************************
    @staticmethod
    def parseXGBtrees(filename):
        splitter = np.zeros((1, 2))
        f = open(filename)
        line = f.readline()
        mdl = []
        flg = False
        while line:
            if 'booster' in line:
                if flg:
                    s = DefragModel.__parseXGBsub(mdl[1:])
                    splitter = np.r_[splitter, s]
                mdl = []
                flg = True
            mdl.append([line.count('\t'), line])
            line = f.readline()
        f.close()
        return DefragModel.__uniqueRows(splitter[1:, :])
    
    @staticmethod
    def __parseXGBsub(mdl):
        splitter = []
        for line in range(len(mdl)):
            if 'leaf' in mdl[line][1]:
                continue
            idx1 = mdl[line][1].find('[f') + 2
            idx2 = mdl[line][1].find('<')
            idx3 = mdl[line][1].find(']')
            v = int(mdl[line][1][idx1:idx2])
            t = float(mdl[line][1][idx2+1:idx3])
            splitter.append((v, t))
        return np.array(splitter)
    
    @staticmethod
    def parseRtrees(dirname):
        splitter = []
        filenames = glob.glob(dirname + "*")
        for filename in filenames:
            df = pd.read_csv(filename, sep="\s+", skiprows=1, header=None)
            for i in range(df.shape[0]):
                v = int(df.ix[i, 3] - 1)
                if v < 0:
                    continue
                t = df.ix[i, 4]
                splitter.append((v, t))
        return DefragModel.__uniqueRows(np.array(splitter))
    
    @staticmethod
    def __uniqueRows(X):
        B = X[np.lexsort(X.T)]
        idx = np.r_[True, np.any(B[1:]!=B[:-1], axis=tuple(range(1,X.ndim)))]
        Z = B[idx]
        return Z

#************************
# inTree Class
#************************
class inTreeModel(RuleModel):
    def __init__(self, modeltype='regression'):
        super().__init__(modeltype=modeltype)
    
    #************************
    # Fit and Related Methods
    #************************
    def fit(self, y, X, filename, featurename=[]):
        self.dim_ = X.shape[1]
        self.setfeaturename(featurename)
        self.setdefaultpred(y)
        if self.modeltype_ == 'regression':
            v1 = np.percentile(y, 17)
            v2 = np.percentile(y, 50)
            v3 = np.percentile(y, 83)
            val = (v1, v2, v3)
        mdl = self.__parsInTreesFile(filename)
        for m in mdl:
            if m[3] == 'X[,1]==X[,1]':
                self.rule_.append([])
            else:
                subrule = []
                ll = m[3].split(' & ')
                for r in ll:
                    id1 = r.find(',') + 1
                    id2 = r.find(']')
                    idx = int(r[id1:id2])
                    if '>' in r:
                        v = 1
                        id1 = r.find('>') + 1
                        t = float(r[id1:])
                    else:
                        v = 0
                        id1 = r.find('<=') + 2
                        t = float(r[id1:])
                    subrule.append((idx, v, t))
                self.rule_.append(subrule)
            if self.modeltype_ == 'classification':
                self.pred_.append(int(m[4]))
            elif self.modeltype_ == 'regression':
                if m[4] == 'L1':
                    self.pred_.append(val[0])
                elif m[4] == 'L2':
                    self.pred_.append(val[1])
                elif m[4] == 'L3':
                    self.pred_.append(val[2])
        
    def __parsInTreesFile(self, filename):
        f = open(filename)
        line = f.readline()
        mdl = []
        while line:
            if not'[' in line:
                line = f.readline()
                continue
            id1 = line.find('[') + 1
            id2 = line.find(',')
            idx = int(line[id1:id2])
            if idx > len(mdl):
                mdl.append(re.findall(r'"([^"]*)"', line))
            else:
                mdl[idx-1] += re.findall(r'"([^"]*)"', line)
            line = f.readline()
        f.close()
        return mdl

#************************
# DTree Class
#************************
class DTreeModel(RuleModel):
    def __init__(self, modeltype='regression', max_depth=[None, 2, 4, 6, 8], min_samples_leaf=[5, 10, 20, 30], cv=5):
        super().__init__(modeltype=modeltype)
        self.max_depth_ = max_depth
        self.min_samples_leaf_ = min_samples_leaf
        self.cv_ = cv
        
    #************************
    # Fit and Related Methods
    #************************
    def fit(self, y, X, featurename=[]):
        self.dim_ = X.shape[1]
        self.setfeaturename(featurename)
        self.setdefaultpred(y)
        param_grid = {"max_depth": self.max_depth_, "min_samples_leaf": self.min_samples_leaf_}
        if self.modeltype_ == 'regression':
            mdl = tree.DecisionTreeRegressor()
        elif self.modeltype_ == 'classification':
            mdl = tree.DecisionTreeClassifier()
        grid_search = GridSearchCV(mdl, param_grid=param_grid, cv=self.cv_)
        grid_search.fit(X, y)
        mdl = grid_search.best_estimator_
        self.__parseTree(mdl)
        
    def __parseTree(self, mdl):
        t = mdl.tree_
        m = len(t.value)
        left = t.children_left
        right = t.children_right
        feature = t.feature
        threshold = t.threshold
        value = t.value
        parent = [-1] * m
        ctype = [-1] * m
        for i in range(m):
            if not left[i] == -1:
                parent[left[i]] = i
                ctype[left[i]] = 0
            if not right[i] == -1:
                parent[right[i]] = i
                ctype[right[i]] = 1
        for i in range(m):
            if not left[i] == -1:
                continue
            subrule = []
            c = ctype[i]
            idx = parent[i]
            while not idx == -1:
                subrule.append((feature[idx], c, threshold[idx]))
                c = ctype[idx]
                idx = parent[idx]
            self.rule_.append(subrule)
            if np.array(value[i]).size > 1:
                self.pred_.append(np.argmax(np.array(value[i])))
            else:
                self.pred_.append(np.asscalar(value[i]))
        