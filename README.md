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
* R: randomForest, inTrees

To run example codes in ``example`` directory:

* Python: xgboost, Scikit-learn
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

In ``example`` directory:

```
python example_data.py
python example_xgb.py
```

When interepreting XGBoost model, use ``parseXGBtrees`` of ``DefragModel`` to get ``splitter``:

```python
bst.dump_model('xgbmodel.txt') # save XGBoost model as text
splitter = DefragModel.parseXGBtrees('./xgbmodel.txt') # parse XGB model into the array of (feature index, threshold)
```
The result of ``python example_xgb.py`` would be someting like this:

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

### Example 3 - Interpreting R [randomForest](https://cran.r-project.org/web/packages/randomForest/randomForest.pdf) Model

In ``example`` directory:

```
python example_data.py
python example_R.py
```

When interepreting R randomForest model, use ``parseRtrees`` of ``DefragModel`` to get ``splitter``:

* In R, save trees into the ``./forest/`` directory. (Remark: The directory has to include only tree text files but not other files.)

```R
ntree = 10
rf <- randomForest(X, y, ntree=ntree) # fit random forest
for (t in 1:ntree) {
    out <- capture.output(getTree(rf, k=t))
    cat(out,file=sprintf('./forest/tree%03d.txt', t),sep="\n",append=FALSE) # save tree
}
```

* In Python, use ``parseRtrees`` to get ``splitter`` from the directory ``./forest/``.

```python
splitter = DefragModel.parseRtrees('./forest/') # parse R trees in ./forest/ into the array of (feature index, threshold)
```

The result of ``python example_R.py`` would be someting like this:


```
<< defragTrees >>
----- Evaluated Results -----
Test Error = 0.014062
Test Coverage = 0.994000

----- Found Rules -----
[Rule  1]
y = -0.004994 when
	 x_1 >= 0.501874
	 x_2 >= 0.501691

[Rule  2]
y = 0.008739 when
	 x_1 < 0.497157
	 x_2 < 0.500868

[Rule  3]
y = 0.991090 when
	 x_1 >= 0.503451
	 x_2 < 0.500868

[Rule  4]
y = 0.997496 when
	 x_1 < 0.500263
	 x_2 >= 0.502623

[Otherwise]
y = 0.512809
```

### Example 4 - Interpreting Scikit-learn Model

In ``example`` directory:

```
python example_data.py
python example_sklearn.py
```

When interepreting Scikit-learn tree ensembles, use ``parseSLtrees`` of ``DefragModel`` to get ``splitter``:


```python
forest = GradientBoostingRegressor(min_samples_leaf=10)
#forest = RandomForestRegressor(min_samples_leaf=10)
#forest = ExtraTreesRegressor(min_samples_leaf=10)
#forest = AdaBoostRegressor()
forest.fit(X, y)
splitter = DefragModel.parseSLtrees(forest) # parse sklearn tree ensembles into the array of (feature index, threshold)
```


The result of ``python example_sklearn.py`` would be someting like this:


```
<< defragTrees >>
----- Evaluated Results -----
Test Error = 0.012797
Test Coverage = 0.993000

----- Found Rules -----
[Rule  1]
y = -0.004994 when
         x_1 >= 0.501874
         x_2 >= 0.501204

[Rule  2]
y = 0.007650 when
         0.221089 <= x_1 < 0.497388
         0.013755 <= x_2 < 0.497672

[Rule  3]
y = 0.009781 when
         x_1 < 0.337727
         x_2 < 0.500619

[Rule  4]
y = 0.991090 when
         x_1 >= 0.504610
         x_2 < 0.499463

[Rule  5]
y = 0.997496 when
         x_1 < 0.500264
         x_2 >= 0.501204

[Otherwise]
y = 0.512809
```



