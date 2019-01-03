### Tune Parameters for the Leaf-wise (Best-first) Tree

**num_leaves**. This is the main parameter to control the complexity of the tree model. Theoretically, we can set num_leaves = 2^(max_depth) to obtain the same number of leaves as depth-wise tree. However, this simple conversion is not good in practice. The reason is that a leaf-wise tree is typically much deeper than a depth-wise tree for a fixed number of leaves. Unconstrained depth can induce over-fitting. Thus, when trying to tune the num_leaves, we should let it be smaller than 2^(max_depth). For example, when the max_depth=7 the depth-wise tree can get good accuracy, but setting num_leaves to 127 may cause over-fitting, and setting it to 70 or 80 may get better accuracy than depth-wise.

**min_data_in_leaf**. This is a very important parameter to prevent over-fitting in a leaf-wise tree. Its optimal value depends on the number of training samples and num_leaves. Setting it to a large value can avoid growing too deep a tree, but may cause under-fitting. In practice, setting it to hundreds or thousands is enough for a large dataset.

**max_depth**. You also can use max_depth to limit the tree depth explicitly.


### For Faster Speed
Use bagging by setting **bagging_fraction** and **bagging_freq**
Use feature sub-sampling by setting **feature_fraction**
Use small **max_bin**
Use save_binary to speed up data loading in future learning
Use parallel learning, refer to Parallel Learning Guide

### For Better Accuracy
Use large **max_bin** (may be slower)
Use small **learning_rate** with large **num_iterations**
Use large **num_leaves** (may cause over-fitting)
Use bigger training data
Try **dart**

### Deal with Over-fitting
Use small **max_bin**
Use small **num_leaves**
Use **min_data_in_leaf** and **min_sum_hessian_in_leaf**
Use bagging by set bagging_fraction and **bagging_freq**
Use feature sub-sampling by set **feature_fraction**
Use bigger training data
Try **lambda_l1**, **lambda_l2** and **min_gain_to_split** for regularization
Try **max_depth** to avoid growing deep tree


### Other params to tune
**max_cat_group**: When the number of category is large, finding the split point on it is easily over-fitting. So LightGBM merges them into ‘max_cat_group’ groups, and finds the split points on the group boundaries, default:64
**min_gain_to_split**: This parameter will describe the minimum gain to make a split. It can used to control number of useful splits in tree



