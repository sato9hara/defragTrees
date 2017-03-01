# -*- coding: utf-8 -*-
"""
@author: Satoshi Hara
"""

import numpy as np
import pandas as pd
from collections import Counter
import glob

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colorbar as colorbar
import pylab as pl

class RTree(object):
    def __init__(self, modeltype='regression'):
        self.modeltype_ = modeltype
        self.left_ = []
        self.right_ = []
        self.index_ = []
        self.threshold_ = []
        self.pred_ = []
    
    def fit(self, filename):
        df = pd.read_csv(filename, sep="\s+", skiprows=1, header=None)
        for i in range(df.shape[0]):
            if not int(df.ix[i, 5]) == -1:
                left = int(df.ix[i, 1]) - 1
                right = int(df.ix[i, 2]) - 1
                index = int(df.ix[i, 3]) - 1
                threshold = df.ix[i, 4]
                pred = np.nan
            else:
                left = -1
                right = -1
                index = -1
                threshold = 0
                if self.modeltype_ == 'regression':
                    pred = df.ix[i, 6]
                elif self.modeltype_ == 'classification':
                    pred = int(df.ix[i, 6]) - 1
            self.left_.append(left)
            self.right_.append(right)
            self.index_.append(index)
            self.threshold_.append(threshold)
            self.pred_.append(pred)
    
    def predict(self, X):
        y = np.zeros(X.shape[0])
        for n in range(X.shape[0]):
            i = 0
            while True:
                if self.left_[i] == -1:
                    y[n] = self.pred_[i]
                    break
                else:
                    if X[n, self.index_[i]] <= self.threshold_[i]:
                        i = self.left_[i]
                    else:
                        i = self.right_[i]
        if self.modeltype_ == 'classification':
            y = y.astype(np.int64)
        return y

class RForest(object):
    def __init__(self, modeltype='regression'):
        self.modeltype_ = modeltype
        self.trees_ = []
    
    def fit(self, dirname):
        self.trees_ = []
        filenames = glob.glob(dirname + "*")
        for filename in filenames:
            tree = RTree(modeltype=self.modeltype_)
            tree.fit(filename)
            self.trees_.append(tree)
    
    def predict(self, X):
        Z = []
        for tree in self.trees_:
            Z.append(tree.predict(X))
        Z = np.array(Z).T
        if self.modeltype_ == 'regression':
            y = np.mean(Z, axis=1)
        elif self.modeltype_ == 'classification':
            y = np.zeros(X.shape[0])
            for n in range(X.shape[0]):
                key = list(Counter(Z[n, :]).keys())
                count = np.array(list(Counter(Z[n, :]).values()))
                idx = np.argmax(count)
                y[n] = key[idx]
            y = y.astype(np.int64)
        return y
        
    def plot(self, X, d1, d2, alpha=0.8, treenum=5, filename=None, box0=None, plot_line=[]):
        
        # boxes
        num = X.shape[0]
        dim = X.shape[1]
        if box0 is None:
            box0 = np.zeros((2, dim))
            for d in range(dim):
                box0[0, d] = np.min(X[:, d])
                box0[1, d] = np.max(X[:, d])
        box0 = box0.astype(np.float64)
        boxes = []
        for n in range(num):
            box = box0.copy()
            for j, tree in enumerate(self.trees_):
                if j >= treenum:
                    break
                i = 0
                while True:
                    if tree.left_[i] == -1:
                        break
                    else:
                        if X[n, tree.index_[i]] <= tree.threshold_[i]:
                            box[1, tree.index_[i]] = min(box[1, tree.index_[i]], tree.threshold_[i])
                            i = tree.left_[i]
                        else:
                            box[0, tree.index_[i]] = max(box[0, tree.index_[i]], tree.threshold_[i])
                            i = tree.right_[i]
            flg = True
            for b in boxes:
                flg *= np.max(np.abs(b[1] - box)) > 1e-8
            if flg:
                pred = self.predict(X[n, :][np.newaxis, :])[0]
                boxes.append((pred, box))
        n = len(boxes)
        
        # plot
        cmap = cm.get_cmap('cool')
        fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw = {'width_ratios':[19, 1]})
        for b in boxes:
            pred = b[0]
            box = b[1]
            c = cmap(1.0 * pred)
            ax1.add_patch(pl.Rectangle(xy=[box[0, d1], box[0, d2]], width=(box[1, d1] - box[0, d1]), height=(box[1, d2] - box[0, d2]), facecolor=c, linewidth='1.0', alpha=alpha))
        if len(plot_line) > 0:
            ax1.plot(plot_line[0], plot_line[1], 'k--')
        ax1.set_xlabel('x1', size=22)
        ax1.set_ylabel('x2', size=22)
        ax1.set_title('Learned Ensemble', size=28)
        colorbar.ColorbarBase(ax2, cmap=cmap, format='%.1f')
        ax2.set_ylabel('Predictor y', size=22)
        plt.show()
        if not filename is None:
            plt.savefig(filename.replace('.pdf', '_%d.pdf' % (n,)), format="pdf", bbox_inches="tight")
            plt.close()
