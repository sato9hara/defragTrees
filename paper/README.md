# Replicating Paper Results

## Preparation

* Download data files from [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/).
* Put data files in ``data`` directory.

## FAB Inference vs. EM Algorithm

Some of the experiments require a few hours of computations.
Recommended to run on a proper machine.

```
python ./tests/paper_synthetic1.py > result_synthetic1.txt
python ./tests/paper_synthetic2.py > result_synthetic1.txt
```

## Comparision with the Baseline Methods

This requires ten parallel computaions.
Some of the experiments require a few hours of computations.
Recommended to run on a proper machine.

```
python ./tests/paper_synthetic1_itr.py
python ./tests/paper_synthetic2_itr.py
```
