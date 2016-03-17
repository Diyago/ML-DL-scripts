from __future__ import division

import numpy as np
from scipy import linalg

from sklearn.linear_model import (RandomizedLasso, lasso_stability_path,
                                  LassoLarsCV)
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import auc, precision_recall_curve
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.utils.extmath import pinvh
from sklearn.utils import ConvergenceWarning
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
from sklearn.feature_selection import chi2
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import random
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import preprocessing

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output

def mutual_incoherence(X_relevant, X_irelevant):
    """Mutual incoherence, as defined by formula (26a) of [Wainwright2006].
    """
    projector = np.dot(np.dot(X_irelevant.T, X_relevant),
                       pinvh(np.dot(X_relevant.T, X_relevant)))
    return np.max(np.abs(projector).sum(axis=1))



# The iris dataset
df_train = pd.read_csv("C:\\Users\\SBT-Ashrapov-IR\\Desktop\\docs\\apps\\CustomerSatisfaction\\data\\train.csv")
df_test = pd.read_csv('C:\\Users\\SBT-Ashrapov-IR\\Desktop\\docs\\apps\\CustomerSatisfaction\\data\\test.csv')



remove = []
# deletening duplicates cols
for col in df_train.columns:
    if df_train[col].std() == 0:
        remove.append(col)

df_train.drop(remove, axis=1, inplace=True)
df_test.drop(remove, axis=1, inplace=True)

# normalize the data attributes
df_train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
df_test.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))



size = len(df_train.columns) -1 

# remove duplicated columns
remove = []
listZeroes = []
remove3 = []

c = df_train.columns
k = 1
for i in range(len(c)-1):
    v = df_train[c[i]].values
    for j in range(i+1,len(c)):
        if np.array_equal(v,df_train[c[j]].values):
            remove.append(c[j])
df_train.drop(remove, axis=1, inplace=True)
df_test.drop(remove, axis=1, inplace=True)            


for i in range(len(df_train)):
        if  df_train["TARGET"][i] == 0:
            listZeroes.append(i)
        if  df_train["TARGET"][i] == 1:
            remove3.append(i)

            
#print listZeroes
print len(df_train)

#print df_train.index
listZeroes = random.sample(listZeroes, len(df_train) - 3007- len(remove3))
remove3 = random.sample(remove3, len(remove3) - 3007)
listZeroes = listZeroes +remove3

df_ones = df_train.iloc[remove3]
for i in xrange(1, 24,1):
    df_train = df_train.append(df_ones)

# df_train.drop(df_train.index[listZeroes],inplace=True)



X = df_train[df_train.columns[0:size]]
y = df_train[[len(df_train.columns)-1]]



test_frac = 0.10
count = len(X_full)
X = X_full.iloc[-int(count*test_frac):]
y = y_full.iloc[-int(count*test_frac):]
  
print "X", X.shape , "y", y.shape
  
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
  
X_new = model.transform(X)

forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
# print "X_new.shape", X_new.shape
# print X_new
