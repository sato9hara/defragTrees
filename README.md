# defragTrees
Python code for tree ensemble interpretation proposed in the following paper.

* S. Hara, K. Hayashi, [Making Tree Ensembles Interpretable: A Bayesian Model Selection Approach](http://arxiv.org/abs/1606.09066), arXiv:1606.09066, 2016.

## Requirements
To use defragTrees:

* Python3.x
* Numpy
* Pandas

To replicate paper results in ``paper`` directory:

* Python: Scikit-learn, Matplotlib, pylab
* R: randomForest, inTrees, nodeHarvest

To run example codes in ``example`` directory:

* Python: XGBoost, Scikit-learn
* R: randomForest

## Usage

Prepare data:

* Input ``X``: feature matrix, numpy array of size (num, dim).
* Output ``y``: output array, numpy array of size (num,).
  * For regression, ``y`` is real value.
  * For classification, ``y`` is class index (i.e., 0, 1, 2, ..., C-1, for C classes).
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
mdl.fit(X, y, splitter, Kmax)
#mdl.fit(X, y, splitter, Kmax, fittype='EM') # use this when one wants exactly Kmax rules to be fitted
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

### Simple Examples
See ``example`` directory.

### Replicating Paper Results
See ``paper`` directory.
