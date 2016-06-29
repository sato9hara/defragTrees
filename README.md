# defragTrees
Python code for tree ensemble interpretation

## Requirements
Python3.x, Numpy, Pandas, Scikit-learn

[For demo_R] R, randomForest, inTrees


## Example
Preparation:

1. (Training Data) Input X, Output y
2. Trained R randomForest information in the directory ./forest/


```
from defragTrees import DefragModel

Kmax = 10
mdl = DefragModel(modeltype=20, maxitr=100, tol=1e-6, restart=20)
splitter = mdl.parseRTrees('./forest/')
mdl.fit(y, X, splitter, Kmax, fittype='FAB')
print(mdl)
```