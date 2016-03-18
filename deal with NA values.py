
import pandas as pd

import numpy as np
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import ensemble
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split

print('Loading data...')
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
#train = pd.read_csv("C://Users//SBT-Ashrapov-IR//Desktop//docs//apps//BNPParibasCardifClaimsManagement//train.csv")
#test = pd.read_csv("C://Users//SBT-Ashrapov-IR//Desktop//docs//apps//BNPParibasCardifClaimsManagement//test.csv")

#train = train.drop(['ID','target','v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)

#test = test.drop(['ID','v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)
id_test = test['ID'].values
test = test.drop(['ID'],axis=1)
target = train['target'].values
train = train.drop(['ID'],axis=1)

print('Clearing...')
for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()):
    if train_series.dtype == 'O':
        #for objects: factorize
        train[train_name], tmp_indexer = pd.factorize(train[train_name])
        test[test_name] = tmp_indexer.get_indexer(test[test_name])
        #but now we have -1 values (NaN)
    else:
        #for int or float: fill NaN
        tmp_len = len(train[train_series.isnull()])
        if tmp_len>0:
            #print "mean", train_series.mean()
            train.loc[train_series.isnull(), train_name] = -999 
        #and Test
        tmp_len = len(test[test_series.isnull()])
        if tmp_len>0:
            test.loc[test_series.isnull(), test_name] = -999

######################################################
# remove constant columns
C = train.columns
eps = 1e-10
dropped_columns = set()
print('Identifing low-variance columns...', end=' ')
for c in C:
    if train[c].var() < eps:
        # print('.. %-30s: too low variance ... column ignored'%(c))
        dropped_columns.add(c)
print('done!')
ignored_columns = ['ID', 'TARGET']

C = list(set(C) - dropped_columns - set(ignored_columns))

# remove duplicate columns
print('Identifying duplicate columns...', end=' ')
for i, c1 in enumerate(C):
    f1 = train[c1].values
    for j, c2 in enumerate(C[i+1:]):
        f2 = train[c2].values
        if np.all(f1 == f2):
            dropped_columns.add(c2)
print('done!')

C = list(set(C) - dropped_columns - set(ignored_columns))
print('# columns dropped: %d'%(len(dropped_columns)))
print('# columns retained: %d'%(len(C)))

train.drop(dropped_columns, axis=1, inplace=True)
test.drop(dropped_columns, axis=1, inplace=True)

######################################################


# # Split the Learning Set
X_fit, X_eval, y_fit, y_eval= train_test_split(
    train, target, test_size=0.2, random_state=1
)

clf = xgb.XGBClassifier(missing=np.nan, max_depth=6, 
                        n_estimators=5, learning_rate=0.15, 
                        subsample=1, colsample_bytree=0.9, seed=1488)

# fitting
clf.fit(X_fit, y_fit, early_stopping_rounds=50, eval_metric="logloss", eval_set=[(X_eval, y_eval)])
#print y_pred
y_pred= clf.predict_proba(test)[:,1]
submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred})


auc_train = mean_squared_error(y_fit, clf.predict_proba(X_fit)[:,1])
auc_valid = mean_squared_error(y_eval, clf.predict_proba(X_eval)[:,1])

print('\n-----------------------')
print('  logloss train: %.5f'%auc_train)
print('  logloss valid: %.5f'%auc_valid)
print('-----------------------')

print('\nModel parameters...')
print('\n-----------------------\n')

