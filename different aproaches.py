
import numpy as np
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM
from pandas.core.frame import DataFrame
from scipy.weave import size_check
from scipy.spatial import distance
#import xgboost as xgb
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.cross_validation import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import StratifiedKFold
from sklearn.cross_validation import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
#example
# np.random.seed(0)
# size = 300
# x = np.random.normal(0, 1, size)
# print "Lower noise", pearsonr(x, x + np.random.normal(0, 1, size))

# load data
df_train = pd.read_csv("C:\\Users\\SBT-Ashrapov-IR\\Desktop\\docs\\apps\\CustomerSatisfaction\\data\\train.csv")
#df_train = pd.read_csv("C:\\Users\\SBT-Ashrapov-IR\\Desktop\\docs\\apps\\CustomerSatisfaction\\data\\train - ench", sep = "\t")
#df_train = pd.read_csv("C:\\Users\\SBT-Ashrapov-IR\\Desktop\\docs\\apps\\CustomerSatisfaction\\data\\test1.csv", sep = "\t") #delim_whitespace=True)
df_test = pd.read_csv('C:\\Users\\SBT-Ashrapov-IR\\Desktop\\docs\\apps\\CustomerSatisfaction\\data\\test.csv')


# remove constant columns
df_train = df_train[[0,2,23,26,28,30,37,52,58,70,71,72,80,85,89,96,110,117,125,126,135,145,153,165,174,222,225,247,248,250,252,267,271,292,293,295,296,297,299,306,307,335,370]]
df_test = df_test[[0,2,23,26,28,30,37,52,58,70,71,72,80,85,89,96,110,117,125,126,135,145,153,165,174,222,225,247,248,250,252,267,271,292,293,295,296,297,299,306,307,335]]


# deletening duplicates cols
remove = []
for col in df_train.columns:
    if df_train[col].std() == 0:
        remove.append(col)

df_train.drop(remove, axis=1, inplace=True)
df_test.drop(remove, axis=1, inplace=True)

size = len(df_train.columns) -1
listToNormalize = []
# normalize the data attributes
for i in xrange (1,size,1):
    listToNormalize.append(df_train.columns[i])
df_train[listToNormalize] = df_train[listToNormalize].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))
df_test[listToNormalize] = df_train[listToNormalize].apply(lambda x: (x - x.mean()) / (x.max() - x.min()))

print df_train

# df_cor = df_train.corr( method = 'pearson', min_periods =1)
# print df_cor[[size]]
num = 0


# toDel = []
# 
# for i in xrange(1,size):
#         #print distance.correlation(df_train[[i]], df_train[[size]])
#         print abs(pearsonr(df_train[[i]], df_train[[size]])[0])
# #         if distance.correlation(df_train[[i]], df_train[[size]]) < 1: 
# #             toDel.append(df_train.columns[i])
#             
print "size of df_train before del", len(df_train.columns)
# df_train.drop(toDel, axis=1,inplace=True)
# df_test.drop(toDel, axis=1,inplace=True)
# print "size of df_train after del",len(df_train.columns)


# remove duplicated columns
remove = []
listZeroes = []
listOnes = []

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
            listOnes.append(i)
           

#print df_train.index
listZeroes = random.sample(listZeroes, len(df_train) - 3007- len(listOnes))
listOnes = random.sample(listOnes, len(listOnes) - 3007)
listZeroes = listZeroes + listOnes
df_train.drop(df_train.index[listZeroes],inplace=True)

# df_ones = df_train.iloc[listOnes]
# for i in xrange(1, 28,1):
#     df_train = df_train.append(df_ones)

print "df_train.shape after append", df_train.shape
                            
print len(df_train)
test_frac = 0.1

count = len(df_train)
train = df_train.iloc[:-int(count*test_frac)]
test = df_train.iloc[-int(count*test_frac):]

y_train = train['TARGET'].values
x_train = train.drop(['ID','TARGET'], axis=1).values

y_test = test['TARGET'].values
x_test = test.drop(['ID','TARGET'], axis=1).values

# length of dataset
len_train = len(x_train)
len_test  = len(x_test)

print('Data is prepared for predicting!')

def get_quality(preds, answers):
    return (sum([1 if pred == ans else 0 for pred, ans in zip(preds, answers)]) / (float)(len(preds)))

y_train = train['TARGET'].values
x_train = train.drop(['ID','TARGET'], axis=1).values

y_test = test['TARGET'].values
x_test = test.drop(['ID','TARGET'], axis=1).values

X_final_test = df_test.drop(['ID'], axis=1).values
df_test.drop(remove, axis=1, inplace=True)
id_final_test = df_test['ID']


#####################################################################################
cv = StratifiedKFold(y_train, n_folds=3, shuffle=True, random_state=1)
 
alg_ngbh = KNeighborsClassifier(n_neighbors=3)
scores = cross_val_score(alg_ngbh, x_train, y_train, cv=cv)
print("Accuracy (k-neighbors): {}+/-{}".format(scores.mean(), scores.std()))
 
from sklearn import linear_model
alg_sgd = linear_model.SGDClassifier(random_state=1)
scores = cross_val_score(alg_sgd, x_train, y_train, cv=cv)
print("Accuracy (sgd): {}+/-{}".format(scores.mean(), scores.std()))
 
from sklearn.svm import SVC
alg_svm = SVC(C=1.0)
scores = cross_val_score(alg_svm, x_train, y_train, cv=cv)
print("Accuracy (svm): {}/{}".format(scores.mean(), scores.std()))
 
from sklearn.naive_bayes import GaussianNB
alg_nbs = GaussianNB()
scores = cross_val_score(alg_nbs, x_train, y_train, cv=cv)
print("Accuracy (naive bayes): {}/{}".format(scores.mean(), scores.std()))
 
def linear_scorer(estimator, x, y):
    scorer_predictions = estimator.predict(x)
 
    scorer_predictions[scorer_predictions > 0.5] = 1
    scorer_predictions[scorer_predictions <= 0.5] = 0
 
    return metrics.accuracy_score(y, scorer_predictions)
 
from sklearn.linear_model import LinearRegression
alg_lnr = LinearRegression()
scores = cross_val_score(alg_lnr, x_train, y_train, cv=cv, 
                         scoring=linear_scorer)
print("Accuracy (linear regression): {}/{}".format(scores.mean(), scores.std()))
 
alg_log = LogisticRegression(random_state=1)
scores = cross_val_score(alg_log, x_train, y_train, cv=cv, 
                         scoring=linear_scorer)
print("Accuracy (logistic regression): {}/{}".format(scores.mean(), scores.std()))
 
alg_frst = RandomForestClassifier(random_state=1, n_estimators=500, min_samples_split=8, min_samples_leaf=2)
scores = cross_val_score(alg_frst, x_train, y_train, cv=cv)
print("Accuracy (random forest): {}/{}".format(scores.mean(), scores.std()))
# 
# 
# Accuracy (k-neighbors): 0.953257911962+/-0.00198568775757
# Accuracy (sgd): 0.960741325973+/-1.94652171974e-05
# Accuracy (svm): 0.960741325973/1.94652171974e-05
# Accuracy (naive bayes): 0.0664737824966/0.00442067324553
# Accuracy (linear regression): 0.960624401524/0.000127715817634
# Accuracy (logistic regression): 0.96062439832/3.43921133709e-05
# Accuracy (random forest): 0.960755941369/1.40500919606e-06
#####################################################################################
alg_frst_model = RandomForestClassifier(random_state=1)
alg_frst_params = [{
    "n_estimators": [   500, 600, 700, 1000],
    "max_depth": [10, 12, 15, 19]
    #"min_samples_leaf": [1, 2, 4]
}]
alg_frst_grid = GridSearchCV(alg_frst_model, alg_frst_params, cv=cv, refit=True, verbose=1, scoring = 'f1')
alg_frst_grid.fit(x_train, y_train)
alg_frst_best = alg_frst_grid.best_estimator_
print("Accuracy (random forest auto): {} with params {}"
       .format(alg_frst_grid.best_score_, alg_frst_grid.best_params_))
  
#Accuracy (random forest auto): 0.960755941419 with params {'min_samples_split': 6, 'n_estimators': 350, 'min_samples_leaf': 2}


#####################################################################################

print "LogisticRegression"
model = LogisticRegression()
model.fit(x_train, y_train)
preds3 = model.predict(x_test)
    
model = LogisticRegression()
model.fit(x_train, y_train)
preds3 = model.predict(x_test)
      
print "quality", get_quality(preds3, y_test) 
preds_final = model.predict(X_final_test)
    
submission = pd.DataFrame({"ID":id_final_test, "TARGET":preds_final})
submission.to_csv("submission_logistic.csv", index=False)
preds_final = model.predict(x_test)
print "precision_score", precision_score(y_test, preds_final)
print "recall_score", recall_score(y_test, preds_final) 
    
d = submission.groupby(["TARGET"]).size()
print d
    
######################################################################################
#    
print "RandomForestClassifier"
clf = RandomForestClassifier(n_estimators=600,  max_depth=12)
scores = cross_validation.cross_val_score(clf, x_train, y_train, cv=3, scoring = 'f1') 
print("Accuracy f1 (random forest): ", scores, scores.std())   
      
clf.fit(x_train, y_train)
preds_final = clf.predict(X_final_test)
submission = pd.DataFrame({"ID":id_final_test, "TARGET":preds_final})
submission.to_csv("submission_tree.csv", index=False)
preds_final = clf.predict(x_test)
print "precision_score", precision_score(y_test, preds_final)
print "recall_score", recall_score(y_test, preds_final)
print submission.groupby(["TARGET"]).size()
  
######################################################################################
from sklearn.ensemble import AdaBoostClassifier #For Classification
from sklearn.tree import DecisionTreeClassifier
 
print "Ada Boost"
dt = DecisionTreeClassifier() 
clf = AdaBoostClassifier(n_estimators=100, base_estimator=dt,learning_rate=1)
clf.fit(x_train,y_train)
scores = cross_validation.cross_val_score(clf, x_train, y_train, cv=cv, scoring = 'f1')
print("Accuracy f1 (Ada Boost): {}/{}".format(scores.mean(), scores.std()))    
preds_final = clf.predict(X_final_test)
submission = pd.DataFrame({"ID":id_final_test, "TARGET":preds_final})
submission.to_csv("submission_AdaBoost.csv", index=False)
preds_final = clf.predict(x_test)
print "precision_score", precision_score(y_test, preds_final)
print "recall_score", recall_score(y_test, preds_final)
print submission.groupby(["TARGET"]).size()
#####################################################################################
  
  
print "GradientBoostingClassifier"
from sklearn.ensemble import GradientBoostingClassifier 
from sklearn.datasets import make_hastie_10_2 
from sklearn.cross_validation import cross_val_score
#X,y = make_hastie_10_2(n_samples = 10000) 
est = GradientBoostingClassifier(n_estimators = 220, max_depth = 13)
est.fit( x_train, y_train) 
pred = est.predict(x_test) 
print est.predict_proba(x_test)[0] 
# scores = cross_val_score(est, x_test, y_test) 
# print "scores.mean", scores.mean() 
print "accuracy", get_quality(pred, y_test) 
   
preds_final = est.predict(X_final_test) 
submission = pd.DataFrame({"ID":id_final_test, "TARGET":preds_final}) 
submission.to_csv("submission_boosting.csv", index=False)
print "precision_score", precision_score(y_test, pred)
print "recall_score", recall_score(y_test, pred) 
print submission.groupby(["TARGET"]).size()
