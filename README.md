# defragTrees
Python code for tree ensemble interpretation proposed in the following paper.

* S. Hara, K. Hayashi, [Making Tree Ensembles Interpretable: A Bayesian Model Selection Approach](http://arxiv.org/abs/1606.09066), arXiv:1606.09066, 2016.

## Requirements
To use defragTrees:

* Python3.x
* Numpy
* Pandas
* Pylab
* Matplotlib

To replicate paper results:

* Python: Scikit-learn
* R: randomForest, inTrees

## Usage

Prepare data:

* Input ``X``: feature matrix, numpy array of size (num, dim).
* Output ``y``: output array, numpy array of size (num,).
* Splitter ``splitter``: thresholds of tree ensembles, numpy array of size (# of split rules, 2).
  * Each row of ``splitter`` is (feature index, threshold). Suppose the split rule is ``second feature < 0.5``, the row of ``splitter`` is then (1, 0.5).

Import the class:

```python
from defragTrees import DefragModel
```

Fit the simplified model:


```python
Kmax = 10 # uppder-bound number of rules to be fitted
mdl = DefragModel(modeltype='regression') # change to 'classification' if necessary.
mdl.fit(y, X, splitter, Kmax)
#mdl.fit(y, X, splitter, Kmax, fittype='EM') # use this when one wants exactly Kmax rules to be fitted
```

Check the learned rules:

```python
print(mdl)
```

For further deitals, see ``defragTrees.py``.
In IPython, one can check:

```python
import defragTrees
defragTrees?
```

## Examples

### Example 1 - Replicating Paper Results

In ``paper`` directory:

```
python paper_synthetic.py > synthetic.txt
python paper_energy.py > energy.txt
python paper_higgs.py > higgs.txt
```

### Example 2 - Interpreting [XGBoost](http://xgboost.readthedocs.io/en/latest/python/python_intro.html) Model

Data Preparation:

```python:exmaple_data.py
import numpy as np

# setting
seed = 1
num = 1000
dim = 2

# data - train
np.random.seed(seed)
Xtr = np.random.rand(num, dim)
ytr = np.zeros(num)
ytr[(Xtr[:, 0] < 0.5) * (Xtr[:, 1] < 0.5)] = 0
ytr[(Xtr[:, 0] >= 0.5) * (Xtr[:, 1] < 0.5)] = 1
ytr[(Xtr[:, 0] < 0.5) * (Xtr[:, 1] >= 0.5)] = 1
ytr[(Xtr[:, 0] >= 0.5) * (Xtr[:, 1] >= 0.5)] = 0
ytr += 0.1 * np.random.randn(num)

# data - test
Xte = np.random.rand(num, dim)
yte = np.zeros(num)
yte[(Xte[:, 0] < 0.5) * (Xte[:, 1] < 0.5)] = 0
yte[(Xte[:, 0] >= 0.5) * (Xte[:, 1] < 0.5)] = 1
yte[(Xte[:, 0] < 0.5) * (Xte[:, 1] >= 0.5)] = 1
yte[(Xte[:, 0] >= 0.5) * (Xte[:, 1] >= 0.5)] = 0
yte += 0.1 * np.random.randn(num)

# save
np.savetxt('./train.csv', np.c_[Xtr, ytr], delimiter=',')
np.savetxt('./test.csv', np.c_[Xte, yte], delimiter=',')
```
XGBoost Model Interpretation:

```python:exmaple_xgb.py
import sys
sys.path.append('../')

import numpy as np
import xgboost as xgb
from defragTrees import DefragModel

# load data
Ztr = np.loadtxt('./train.csv', delimiter=',')
Zte = np.loadtxt('./test.csv', delimiter=',')
Xtr = Ztr[:, :-1]
ytr = Ztr[:, -1]
Xte = Zte[:, :-1]
yte = Zte[:, -1]

# train xgboost
num_round = 20
dtrain = xgb.DMatrix(Xtr, label=ytr)
param = {'max_depth':4, 'eta':0.3, 'silent':1, 'objective':'reg:linear'}
bst = xgb.train(param, dtrain, num_round)

# output xgb model as text
bst.dump_model('xgbmodel.txt')

# fit simplified model
Kmax = 10
mdl = DefragModel(modeltype='regression', maxitr=100, tol=1e-6, restart=20, verbose=0)
splitter = mdl.parseXGBtrees('./xgbmodel.txt') # parse XGB model into the array of (feature index, threshold)
mdl.fit(ytr, Xtr, splitter, Kmax, fittype='FAB')

# results
score, cover = mdl.evaluate(yte, Xte)
print()
print('<< defragTrees >>')
print('----- Evaluated Results -----')
print('Test Error = %f' % (score,))
print('Test Coverage = %f' % (cover,))
print()
print('----- Found Rules -----')
print(mdl)
```
The result would be someting like this:

```
<< defragTrees >>
----- Evaluated Results -----
Test Error = 0.014599
Test Coverage = 0.995000

----- Found Rules -----
[Rule  1]
y = -0.004994 when
         x_1 >= 0.501874
         x_2 >= 0.502093

[Rule  2]
y = 0.008739 when
         x_1 < 0.497388
         x_2 < 0.500868

[Rule  3]
y = 0.991090 when
         x_1 >= 0.503257
         x_2 < 0.500868

[Rule  4]
y = 0.997496 when
         x_1 < 0.501469
         x_2 >= 0.502093

[Otherwise]
y = 0.512809
```