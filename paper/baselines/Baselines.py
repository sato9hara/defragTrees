# -*- coding: utf-8 -*-
"""
@author: Satoshi Hara
"""

import sys
sys.path.append('../')
from defragTrees import *
import BATree
from RForest import RForest

import numpy as np
import re
from sklearn import tree
from sklearn.grid_search import GridSearchCV

#************************
# inTree Class
#************************
class inTreeModel(RuleModel):
    def __init__(self, modeltype='regression'):
        super().__init__(modeltype=modeltype)
    
    #************************
    # Fit and Related Methods
    #************************
    def fit(self, X, y, filename, featurename=[]):
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
        self.weight_ = np.arange(len(self.rule_))[::-1].tolist()
        
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
# NHarvest Class
#************************
class NHarvestModel(RuleModel):
    def __init__(self, modeltype='regression'):
        super().__init__(modeltype=modeltype)
    
    #************************
    # Fit and Related Methods
    #************************
    def fit(self, X, y, filename, featurename=[]):
        self.dim_ = X.shape[1]
        self.setfeaturename(featurename)
        rule, pred, weight = self.__parsNHarvestFile(filename)
        self.setdefaultpred(pred[-1])
        idx = np.argsort(weight[:-1])[::-1]
        self.rule_ = [rule[i] for i in idx]
        if self.modeltype_ == 'regression':
            self.pred_ = [pred[i] for i in idx]
        elif self.modeltype_ == 'classification':
            self.pred_ = (np.array([pred[i] for i in idx]) > 0.5).astype(int).tolist()
        self.weight_ = [weight[i] for i in idx]
    
    def __parsNHarvestFile(self, filename):
        f = open(filename)
        line = f.readline()
        rule = []
        pred = []
        weight = []
        while line:
            f.readline()
            subrule = []
            line = f.readline()
            while (line[0] != 'a'):
                s = line.split()
                idx = int(s[1])
                low = float(s[2])
                up = float(s[3])
                if not np.isinf(low):
                    subrule.append((idx, 1, low))
                if not np.isinf(up):
                    subrule.append((idx, 0, up))
                line = f.readline()
            if (len(subrule) > 0):
                rule.append(subrule)
            while True:
                line = f.readline()
                if (line[0] == 'a'):
                    s = line.split('"')
                    if (s[1] == 'predict'):
                        break
            line = f.readline()
            s = line.split()
            pred.append(float(s[1]))
            f.readline()
            line = f.readline()
            s = line.split()
            weight.append(float(s[1]))
            line = f.readline()
            line = f.readline()
            line = f.readline()
            line = f.readline()
            if not line[:2] == '[[':
                break
        f.close()
        return rule, pred, weight


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
    def fit(self, X, y, featurename=[]):
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
        self.weight_ = np.ones(len(self.rule_))
        
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
                subrule.append((int(feature[idx])+1, c, threshold[idx]))
                c = ctype[idx]
                idx = parent[idx]
            self.rule_.append(subrule)
            if np.array(value[i]).size > 1:
                self.pred_.append(np.argmax(np.array(value[i])))
            else:
                self.pred_.append(np.asscalar(value[i]))

#************************
# BTree Class
#************************
class BTreeModel(RuleModel):
    def __init__(self, modeltype='regression', max_depth=[2, 3, 4, 6, 8, 10], min_samples_leaf=[10], cv=5, smear_num=100, njobs=1, seed=0):
        super().__init__(modeltype=modeltype)
        self.max_depth_ = max_depth
        self.min_samples_leaf_ = min_samples_leaf
        self.cv_ = cv
        self.smear_num_ = smear_num
        self.njobs_ = njobs
        self.seed_ = seed
        
    #************************
    # Fit and Related Methods
    #************************
    def fit(self, X, y, dirname, featurename=[]):
        self.dim_ = X.shape[1]
        self.setfeaturename(featurename)
        self.setdefaultpred(y)
        mdl = RForest(modeltype=self.modeltype_)
        mdl.fit(dirname)
        tree = BATree.fitBATreeCV(X, y, mdl, modeltype=self.modeltype_, max_depth=self.max_depth_, min_samples_split=self.min_samples_leaf_, cv=self.cv_, seed=self.seed_, smear_num=self.smear_num_, njobs=self.njobs_)
        self.__parseTree(tree)
        self.weight_ = np.ones(len(self.rule_))
        return tree
        
    def __parseTree(self, tree):
        m = len(tree.pred_)
        left = tree.left_
        right = tree.right_
        feature = tree.index_
        threshold = tree.threshold_
        value = tree.pred_
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
                subrule.append((int(feature[idx])+1, c, threshold[idx]))
                c = ctype[idx]
                idx = parent[idx]
            self.rule_.append(subrule)
            if np.array(value[i]).size > 1:
                self.pred_.append(np.argmax(np.array(value[i])))
            else:
                self.pred_.append(np.asscalar(value[i]))
