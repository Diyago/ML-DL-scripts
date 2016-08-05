# с очисткой и факторизацией данных
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split

#2--Watch the eta, min_child_weight, subsample, colsample_bytree, max_depth and number of rounds



print('Loading data...')
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

id_test = test['ID'].values
test = test.drop(['ID'],axis=1)
test = test.drop(['v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)
target = train['target'].values
train = train.drop(['ID'],axis=1)
train = train.drop(['target','v8','v23','v25','v31','v36','v37','v46','v51','v53','v54','v63','v73','v75','v79','v81','v82','v89','v92','v95','v105','v107','v108','v109','v110','v116','v117','v118','v119','v123','v124','v128'],axis=1)

#factorizing only train and test at once to have same dictionary for them

shapeTrain = train.shape[0]
shapeTest = test.shape[0]
train = train.append(test)

from sklearn import preprocessing 
for f in train.columns: 
    if train[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder() 
        lbl.fit(list(train[f].values)) 
        train[f] = lbl.transform(list(train[f].values))

test = train[shapeTrain:shapeTrain+shapeTest]
train = train[0:shapeTrain]

# for f in test.columns: 
#     if test[f].dtype=='object': 
#         lbl = preprocessing.LabelEncoder() 
#         lbl.fit(list(test[f].values)) 
#         test[f] = lbl.transform(list(test[f].values))


#xgboost will handle NA values
#train.fillna((-999), inplace=True) 
#test.fillna((-999), inplace=True)

train=np.array(train) 
test=np.array(test) 
train = train.astype(float) 
test = test.astype(float)

######################################################


# # Split the Learning Set
X_fit, X_eval, y_fit, y_eval= train_test_split(
    train, target, test_size=0.15, random_state=1
)


xgtrain = xgb.DMatrix(X_fit, y_fit)
xgtest = xgb.DMatrix(test)

#n_estimators and early_stopping_rounds should be increased
clf = xgb.XGBClassifier(missing=np.nan, max_depth=7, 
                        n_estimators=300, learning_rate=0.05, 
                        subsample=1, colsample_bytree=0.9, seed=2100,objective= 'binary:logistic')

# fitting
clf.fit(X_fit, y_fit, early_stopping_rounds=35,  eval_metric="logloss", eval_set=[(X_eval, y_eval)])

# scores
from  sklearn.metrics import log_loss
log_train = log_loss(y_fit, clf.predict_proba(X_fit)[:,1])
log_valid = log_loss(y_eval, clf.predict_proba(X_eval)[:,1])


print('\n-----------------------')
print('  logloss train: %.5f'%log_train)
print('  logloss valid: %.5f'%log_valid)
print('-----------------------')

print('\nModel parameters...')
print(clf.get_params())


#print y_pred
y_pred= clf.predict_proba(test,ntree_limit=clf.best_ntree_limit)[:,1]
submission = pd.DataFrame({"ID":id_test, "PredictedProb":y_pred})
submission.to_csv("submission.csv", index=False)

print ("Success")
#########################################################





