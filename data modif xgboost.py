# с очисткой и прилизыванием данных
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.cross_validation import train_test_split



print('Loading data...')
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

id_test = test['ID'].values
test = test.drop(['ID'],axis=1)
target = train['target'].values
train = train.drop(['ID'],axis=1)



from sklearn import preprocessing 
for f in train.columns: 
    if train[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder() 
        lbl.fit(list(train[f].values)) 
        train[f] = lbl.transform(list(train[f].values))

for f in test.columns: 
    if test[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder() 
        lbl.fit(list(test[f].values)) 
        test[f] = lbl.transform(list(test[f].values))

train.fillna((-999), inplace=True) 
test.fillna((-999), inplace=True)

train=np.array(train) 
test=np.array(test) 
train = train.astype(float) 
test = test.astype(float)

######################################################


# # Split the Learning Set
X_fit, X_eval, y_fit, y_eval= train_test_split(
    train, target, test_size=0.2, random_state=1
)


xgtrain = xgb.DMatrix(X_fit, y_fit)

xgtest = xgb.DMatrix(test)

clf = xgb.XGBClassifier(missing=np.nan, max_depth=6, 
                        n_estimators=750, learning_rate=0.15, 
                        subsample=1, colsample_bytree=0.9, seed=1488)

# fitting
clf.fit(X_fit, y_fit, early_stopping_rounds=50, eval_metric="logloss", eval_set=[(X_eval, y_eval)])

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
y_pred= clf.predict_proba(test)[:,1]
submission = pd.DataFrame({"ID":id_test, "PredictedProb":y_pred})
submission.to_csv("submission.csv", index=False)

print ("Success")
#########################################################


