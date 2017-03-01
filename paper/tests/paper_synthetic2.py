# -*- coding: utf-8 -*-
"""
@author: Satoshi Hara
"""

import sys
import os
sys.path.append(os.path.abspath('./'))
sys.path.append(os.path.abspath('./baselines/'))
sys.path.append(os.path.abspath('../'))

import numpy as np
import paper_sub
from RForest import RForest

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colorbar as colorbar

def plotTZ(filename=None):
    t = np.linspace(0, 1, 101)
    z = 0.25 + 0.5 / (1 + np.exp(- 20 * (t - 0.5))) + 0.05 * np.cos(t * 2 * np.pi)
    cmap = cm.get_cmap('cool')
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw = {'width_ratios':[19, 1]})
    poly1 = [[0, 0]]
    poly1.extend([[t[i], z[i]] for i in range(t.size)])
    poly1.extend([[1, 0], [0, 0]])
    poly2 = [[0, 1]]
    poly2.extend([[t[i], z[i]] for i in range(t.size)])
    poly2.extend([[1, 1], [0, 1]])
    poly1 = plt.Polygon(poly1,fc=cmap(0.0))
    poly2 = plt.Polygon(poly2,fc=cmap(1.0))
    ax1.add_patch(poly1)
    ax1.add_patch(poly2)
    ax1.set_xlabel('x1', size=22)
    ax1.set_ylabel('x2', size=22)
    ax1.set_title('True Data', size=28)
    colorbar.ColorbarBase(ax2, cmap=cmap, format='%.1f')
    ax2.set_ylabel('Output y', size=22)
    plt.show()
    if not filename is None:
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        plt.close()
    
def plotForest(filename=None):
    forest = RForest(modeltype='classification')
    forest.fit('./result/result_synthetic2/forest/')
    X = np.c_[np.kron(np.linspace(0, 1, 201), np.ones(201)), np.kron(np.ones(201), np.linspace(0, 1, 201))]
    forest.plot(X, 0, 1, box0=np.array([[0.0, 0.0], [1.0, 1.0]]), filename=filename)
    
if __name__ == "__main__":

    # setting
    prefix = 'synthetic2'
    seed = 0
    num = 1000
    dim = 2
    
    # data - boundary
    b = 0.9
    t = np.linspace(0, 1, 101)
    z = 0.25 + 0.5 / (1 + np.exp(- 20 * (t - 0.5))) + 0.05 * np.cos(t * 2 * np.pi)
    
    # data - train
    np.random.seed(seed)
    Xtr = np.random.rand(num, dim)
    ytr = np.zeros(num)
    ytr = (Xtr[:, 1] > 0.25 + 0.5 / (1 + np.exp(- 20 * (Xtr[:, 0] - 0.5))) + 0.05 * np.cos(Xtr[:, 0] * 2 * np.pi))
    ytr = np.logical_xor(ytr, np.random.rand(num) > b)
    
    # data - test
    Xte = np.random.rand(num, dim)
    yte = np.zeros(num)
    yte = (Xte[:, 1] > 0.25 + 0.5 / (1 + np.exp(- 20 * (Xte[:, 0] - 0.5))) + 0.05 * np.cos(Xte[:, 0] * 2 * np.pi))
    yte = np.logical_xor(yte, np.random.rand(num) > b)
    
    # save
    dirname = './result/result_%s' % (prefix,)
    if not os.path.exists('./result/'):
        os.mkdir('./result/')
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    trfile = '%s/%s_train.csv' % (dirname, prefix)
    tefile = '%s/%s_test.csv' % (dirname, prefix)
    np.savetxt(trfile, np.c_[Xtr, ytr], delimiter=',')
    np.savetxt(tefile, np.c_[Xte, yte], delimiter=',')
    
    # demo_R
    Kmax = 10
    restart = 20
    treenum = 100
    M = range(1, 11)
    #paper_sub.run(prefix, Kmax, restart, treenum=treenum, modeltype='classification', plot=True, plot_line=[[t, z]])
    paper_sub.run(prefix, Kmax, restart, treenum=treenum, modeltype='classification', plot=True, plot_line=[[t, z]], M=M, compare=True)
    
    # plot
    plotTZ('%s/%s_true.pdf' % (dirname, prefix))
    plotForest('%s/%s_rf_tree05_seed00.pdf' % (dirname, prefix))
    