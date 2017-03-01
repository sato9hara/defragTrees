# Examples

## Example 1 - Interpreting [XGBoost](http://xgboost.readthedocs.io/en/latest/python/python_intro.html) Model

In this directory:

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
Test Error = 0.092000
Test Coverage = 0.994000
Overlap = 0.994000

----- Found Rules -----
[Rule  1]
y = 0 when
	 x_1 >= 0.496465
	 x_2 >= 0.497393

[Rule  2]
y = 0 when
	 x_1 < 0.493678
	 x_2 < 0.497393

[Rule  3]
y = 1 when
	 x_1 < 0.496097
	 x_2 >= 0.499752

[Rule  4]
y = 1 when
	 x_1 >= 0.500412
	 x_2 < 0.497393

[Otherwise]
y = 1
```

## Example 2 - Interpreting R [randomForest](https://cran.r-project.org/web/packages/randomForest/randomForest.pdf) Model

In this directory:

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
Test Error = 0.092000
Test Coverage = 0.982000
Overlap = 0.982000

----- Found Rules -----
[Rule  1]
y = 0 when
	 x_1 >= 0.501998
	 x_2 >= 0.489962

[Rule  2]
y = 0 when
	 x_1 < 0.491667
	 x_2 < 0.495262

[Rule  3]
y = 1 when
	 x_1 >= 0.500821
	 x_2 < 0.483560

[Rule  4]
y = 1 when
	 x_1 < 0.498655
	 x_2 >= 0.500795

[Otherwise]
y = 1
```

## Example 3 - Interpreting Scikit-learn Model

In this directory:

```
python example_data.py
python example_sklearn.py
```

When interepreting Scikit-learn tree ensembles, use ``parseSLtrees`` of ``DefragModel`` to get ``splitter``:


```python
forest = GradientBoostingClassifier(min_samples_leaf=10)
#forest = RandomForestClassifier(min_samples_leaf=10)
#forest = ExtraTreesClassifier(min_samples_leaf=10)
#forest = AdaBoostClassifier()
forest.fit(Xtr, ytr)
splitter = DefragModel.parseSLtrees(forest) # parse sklearn tree ensembles into the array of (feature index, threshold)
```


The result of ``python example_sklearn.py`` would be someting like this:


```
<< defragTrees >>
----- Evaluated Results -----
Test Error = 0.106000
Test Coverage = 0.992000
Overlap = 1.029000

----- Found Rules -----
[Rule  1]
y = 0 when
	 x_1 < 0.490484
	 x_2 < 0.487602

[Rule  2]
y = 0 when
	 x_1 >= 0.500412
	 x_2 >= 0.505577

[Rule  3]
y = 1 when
	 0.500412 <= x_1 < 0.928521
	 x_2 < 0.504160

[Rule  4]
y = 1 when
	 x_1 < 0.500412
	 x_2 >= 0.487602

[Rule  5]
y = 1 when
	 x_1 >= 0.839603
	 x_2 < 0.484935

[Otherwise]
y = 1
```



