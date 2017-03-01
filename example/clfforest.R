# data
Z <- read.csv('train.csv', header=F)
X <- Z[,1:(ncol(Z)-1)]
y <- as.factor(Z[,ncol(Z)])

# random forest training
library(randomForest)
set.seed(0)
ntree=10
rf <- randomForest(X, y, ntree=ntree, nodesize=5, maxnodes=16)

# save trees
dir.create('./forest', showWarnings = FALSE)
for (t in 1:ntree) {
    out <- capture.output(getTree(rf, k=t))
    cat(out,file=sprintf('./forest/tree%03d.txt', t),sep="\n",append=FALSE)
}