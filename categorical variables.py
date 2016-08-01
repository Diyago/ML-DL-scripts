from sklearn.feature_extraction import DictVectorizer as DV

target = 'TARGET_FLAG'
p_target = 'P_' + target
index = 'INDEX'
skip_feature_columns = [target, 'TARGET_AMT']

def dollar_to_numeric(value):
    """$14,230 -> 14230"""
    if type(value) != str :
        return value
    else :
        return float(value.lstrip('$').replace(',', ''))

import numpy

numeric = []
categorical = []
for column in sorted([column for column in train.columns if column not in skip_feature_columns]) :
    if type(train[column][0] == str) :
        try :
            train[column] = train[column].apply(dollar_to_numeric)
            test[column] = test[column].apply(dollar_to_numeric)
        except ValueError :
            pass
    if type(train[column][0]) in [numpy.float64, numpy.int64] :
        numeric.append(column)
    else :
        categorical.append(column)
        
    print column, train[column].nunique(), type(train[column][0]), train[column][0]
    
numeric, categorical

vectorizer = DV(sparse = False)

X_train_cat = vectorizer.fit_transform(train[categorical].fillna('NA').T.to_dict().values())
X_test_cat = vectorizer.transform(test[categorical].fillna('NA').T.to_dict().values())

X_train = numpy.hstack([train[numeric].fillna(-999).values, X_train_cat])
y_train = train[target]

X_test = numpy.hstack([test[numeric].fillna(-999).values, X_test_cat])
test_ids = test[index]
