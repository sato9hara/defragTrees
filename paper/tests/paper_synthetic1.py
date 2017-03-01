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
import pylab as pl

def plotTZ(filename=None):
    cmap = cm.get_cmap('cool')
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw = {'width_ratios':[19, 1]})
    ax1.add_patch(pl.Rectangle(xy=[0, 0], width=0.5, height=0.5, facecolor=cmap(0.0), linewidth='2.0'))
    ax1.add_patch(pl.Rectangle(xy=[0.5, 0.5], width=0.5, height=0.5, facecolor=cmap(0.0), linewidth='2.0'))
    ax1.add_patch(pl.Rectangle(xy=[0, 0.5], width=0.5, height=0.5, facecolor=cmap(1.0), linewidth='2.0'))
    ax1.add_patch(pl.Rectangle(xy=[0.5, 0], width=0.5, height=0.5, facecolor=cmap(1.0), linewidth='2.0'))
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
    forest.fit('./result/result_synthetic1/forest/')
    X = np.c_[np.kron(np.linspace(0, 1, 201), np.ones(201)), np.kron(np.ones(201), np.linspace(0, 1, 201))]
    forest.plot(X, 0, 1, box0=np.array([[0.0, 0.0], [1.0, 1.0]]), filename=filename)
    
if __name__ == "__main__":

    # setting
    prefix = 'synthetic1'
    seed = 0
    num = 1000
    dim = 2
    
    # data - boundary
    b = 0.9
    t1 = np.array([0.0, 1.0])
    z1 = np.array([0.5, 0.5])
    t2 = np.array([0.5, 0.5])
    z2 = np.array([0.0, 1.0])
    
    # data - train
    np.random.seed(seed)
    Xtr = np.random.rand(num, dim)
    ytr = np.zeros(num)
    ytr = np.logical_xor(Xtr[:, 0] > 0.5, Xtr[:, 1] > 0.5)
    ytr = np.logical_xor(ytr, np.random.rand(num) > b)
    
    # data - test
    Xte = np.random.rand(num, dim)
    yte = np.zeros(num)
    yte = np.logical_xor(Xte[:, 0] > 0.5, Xte[:, 1] > 0.5)
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
    #paper_sub.run(prefix, Kmax, restart, treenum=treenum, modeltype='classification', plot=True, plot_line=[[t1, z1], [t2, z2]])
    paper_sub.run(prefix, Kmax, restart, treenum=treenum, modeltype='classification', plot=True, plot_line=[[t1, z1], [t2, z2]], M=M, compare=True)
    
    # plot
    plotTZ('%s/%s_true.pdf' % (dirname, prefix))
    plotForest('%s/%s_rf_tree05_seed00.pdf' % (dirname, prefix))
    