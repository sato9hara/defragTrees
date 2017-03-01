# Replicating Paper Results

## Requirements

* Python3.x
* Numpy
* Pandas
* Scikit-learn
* Matplotlib
* pylab
* R: randomForest, inTrees, nodeHarvest

## Preparation

* Download data files from [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/).
* Put data files in ``data`` directory.

## FAB Inference vs. EM Algorithm

Some of the experiments require a few hours of computations.
Recommended to run on a proper machine.

```
python ./tests/paper_synthetic1.py > result_synthetic1.txt
python ./tests/paper_synthetic2.py > result_synthetic1.txt
python ./tests/paper_spambase.py > result_spambase.txt
python ./tests/paper_miniboone.py > result_miniboone.txt
python ./tests/paper_higgs.py > result_higgs.txt
python ./tests/paper_energy.py > result_energy.txt
```

## Comparision with the Baseline Methods

Some of the experiments require ten parallel computaions.
Some of the experiments also require a few hours of computations.
Recommended to run on a proper machine.

```
python ./tests/paper_synthetic1_itr.py
python ./tests/paper_synthetic2_itr.py
python ./tests/paper_spambase_itr.py
python ./tests/paper_miniboone_itr.py
python ./tests/paper_higgs_itr.py
python ./tests/paper_energy_itr.py
```
