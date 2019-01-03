Fill reasonable values for key inputs:

**learning_rate**: 0.01; **n_estimators**: 100 if the size of your data is high, 1000 is if it is medium-low; **max_depth**: 3;
**subsample**: 0.8; **colsample_bytree**: 1; **gamma**: 1

Run **model.fit(eval_set, eval_metric)** and diagnose your first run, specifically the n_estimators parameter
Optimize **max_depth** parameter. 

Recommended going from a low **max_depth** (3 for instance) and then increasing it incrementally by 1, and stopping when there’s no performance gain of increasing it. This will help simplify your model and avoid overfitting

Now play around with the learning rate and the features that avoids overfitting:
**learning_rate**: usually between 0.1 and 0.01. If you’re focused on performance and have time in front of you, decrease incrementally the learning rate while increasing the number of trees.

**subsample**, which is for each tree the % of rows taken to build the tree. I recommend not taking out too many rows, as performance will drop a lot. Take values from 0.8 to 1. Typical values: 0.5-1

**colsample_bytree**: number of columns used by each tree. In order to avoid some columns to take too much credit for the prediction (think of it like in recommender systems when you recommend the most purchased products and forget about the long tail), take out a good proportion of columns. Values from 0.3 to 0.8 if you have many columns (especially if you did one-hot encoding), or 0.8 to 1 if you only have a few columns.

**gamma**: usually misunderstood parameter, it acts as a regularization parameter. Either 0, 1 or 5.

**eta** [default=0.3]
Analogous to learning rate in GBM
Makes the model more robust by shrinking the weights on each step
Typical final values to be used: 0.01-0.2

**min_child_weight** [default=1]
Defines the minimum sum of weights of all observations required in a child.
This is similar to min_child_leaf in GBM but not exactly. This refers to min “sum of weights” of observations while GBM has min “number of observations”.
Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
Too high values can lead to under-fitting hence, it should be tuned using CV.

Link https://xgboost.readthedocs.io/en/latest/parameter.html
