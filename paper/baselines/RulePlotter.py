# -*- coding: utf-8 -*-
"""
@author: Satoshi Hara
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colorbar as colorbar
import pylab as pl

def plotRule(mdl, X, d1, d2, alpha=0.8, filename='', rnum=-1, plot_line=[]):
    cmap = cm.get_cmap('cool')
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw = {'width_ratios':[19, 1]})
    if rnum <= 0:
        rnum = len(mdl.rule_)
    else:
        rnum = min(len(mdl.rule_), rnum)
    idx = np.argsort(mdl.weight_[:rnum])
    for i in range(rnum):
        r = mdl.rule_[idx[i]]
        box, vmin, vmax = __r2boxWithX(r, X)
        if mdl.modeltype_ == 'regression':
            c = cmap(mdl.pred_[idx[i]])
        elif mdl.modeltype_ == 'classification':
            r = mdl.pred_[idx[i]] / max(np.unique(mdl.pred_).size - 1, 1)
            c = cmap(r)
        ax1.add_patch(pl.Rectangle(xy=[box[0, d1], box[0, d2]], width=(box[1, d1] - box[0, d1]), height=(box[1, d2] - box[0, d2]), facecolor=c, linewidth='2.0', alpha=alpha))
    if len(plot_line) > 0:
        for l in plot_line:
            ax1.plot(l[0], l[1], 'k--')
    ax1.set_xlabel('x1', size=22)
    ax1.set_ylabel('x2', size=22)
    ax1.set_title('Simplified Model (K = %d)' % (rnum,), size=28)
    colorbar.ColorbarBase(ax2, cmap=cmap, format='%.1f')
    ax2.set_ylabel('Predictor y', size=22)
    plt.show()
    if not filename == '':
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        plt.close()
        
def plotEachRule(mdl, X, d1, d2, alpha=0.8, filename='', rnum=-1, plot_line=[]):
    if rnum <= 0:
        rnum = len(mdl.rule_)
    else:
        rnum = min(len(mdl.rule_), rnum)
    m = rnum // 4
    if m * 4 < rnum:
        m += 1
    cmap = cm.get_cmap('cool')
    fig, ax = plt.subplots(m, 4 + 1, figsize=(4 * 4, 3 * m), gridspec_kw = {'width_ratios':[15, 15, 15, 15, 1]})
    idx = np.argsort(mdl.weight_[:rnum])
    for i in range(rnum):
        j = i // 4
        k = i - 4 * j
        r = mdl.rule_[idx[i]]
        box, vmin, vmax = __r2boxWithX(r, X)
        if mdl.modeltype_ == 'regression':
            c = cmap(mdl.pred_[idx[i]])
        elif mdl.modeltype_ == 'classification':
            r = mdl.pred_[idx[i]] / max(np.unique(mdl.pred_).size - 1, 1)
            c = cmap(r)
        ax[j, k].add_patch(pl.Rectangle(xy=[box[0, d1], box[0, d2]], width=(box[1, d1] - box[0, d1]), height=(box[1, d2] - box[0, d2]), facecolor=c, linewidth='2.0', alpha=alpha))
        if len(plot_line) > 0:
            for l in plot_line:
                ax[j, k].plot(l[0], l[1], 'k--')
        ax[j, k].set_xlim([0, 1])
        ax[j, k].set_ylim([0, 1])
        if k == 3:
            cbar = colorbar.ColorbarBase(ax[j, -1], cmap=cmap, format='%.1f', ticks=[0.0, 0.5, 1.0])
            cbar.ax.set_yticklabels([0.0, 0.5, 1.0])
            ax[j, -1].set_ylabel('Predictor y', size=12)
    plt.show()
    if not filename == '':
        plt.savefig(filename, format="pdf", bbox_inches="tight")
        plt.close()

def __r2boxWithX(r, X):
    vmin = np.min(X, axis=0)
    vmax = np.max(X, axis=0)
    box = np.c_[vmin, vmax].T
    for rr in r:
        if rr[1] == 0:
            box[1, rr[0]-1] = np.minimum(box[1, rr[0]-1], rr[2])
        else:
            box[0, rr[0]-1] = np.maximum(box[0, rr[0]-1], rr[2])
    return box, vmin, vmax        
