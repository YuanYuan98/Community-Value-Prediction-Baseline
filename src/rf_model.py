# -*- coding: utf-8 -*-
"""
@author: zgz
"""

import numpy as np
import copy
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold


def split_train_test(n, nfolds, rnd_state=None):
    rnd_state = np.random.RandomState() if rnd_state is None else rnd_state
    idx = rnd_state.permutation(n)
    idx = idx.tolist()
    stride = int(n/nfolds)
    # 先把idx分成10份
    idx = [idx[i*stride:(i+1)*stride] for i in range(nfolds)]
    train_idx, test_idx = {},{}
    for fold in range(nfolds):
        test_idx[fold] = np.array(copy.deepcopy(idx[fold]))
        train_idx[fold] = []
        for i in range(nfolds):
            if i!=fold:
                train_idx[fold] += idx[i]
        train_idx[fold] = np.array(train_idx[fold])
    return train_idx, test_idx


def load_data(path):
    dataset = np.load(path, allow_pickle=True)
    X = dataset[:,:-1]
    y = dataset[:,-1]
    return X,y

n_folds = 10
data_path = '../data/sample2_dataset_rf.npy'
X,y = load_data(data_path)



### Feature Scaling
# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)

# min_max_scaler = MinMaxScaler(feature_range=(-1,1))
# X_train = min_max_scaler.fit_transform(X_train)
# X_test = min_max_scaler.fit_transform(X_test)



### grid search
# param_test = {'n_estimators':range(170,301,10)}
# param_test = {'max_depth':range(10,17,1), 'min_samples_split':range(10,81,10)}
# param_test = {'min_samples_split':range(10,71,10), 'min_samples_leaf':range(5,15,1)}
# param_test = {'max_features':range(4,12,1)}
# param_test = {'n_estimators':range(30,201,10), 'max_depth':range(3,14,1), 'min_samples_split':range(100,221,10), 
# 'max_features':range(1,5,1), 'min_samples_split':range(100,181,10), 'min_samples_leaf':range(8,12,1), 'max_features':range(1,5,1)}

# gsearch = GridSearchCV(estimator = RandomForestRegressor(oob_score=True, random_state=0, 
#     max_depth=15,
#     min_samples_split=10,
#     min_samples_leaf=3,
#     max_features=11),
#    param_grid = param_test, scoring='neg_mean_absolute_error',iid=False, cv=10)

# gsearch = GridSearchCV(estimator = RandomForestRegressor(oob_score=True, random_state=0), \
#    param_grid = param_test, scoring='neg_mean_squared_error',iid=False, cv=10)

# gsearch.fit(X,y)
# print(gsearch.best_params_, gsearch.best_score_)



### evaluation
regr = RandomForestRegressor(oob_score=True, random_state=0, 
    n_estimators=170,
    max_depth=13,
    min_samples_split=30,
    min_samples_leaf=10,
    max_features=11)

# fold = 0
# random_state = np.random.RandomState(630)
# train_idx, test_idx = split_train_test(1500, 10, random_state)
# train_index = train_idx[fold]
# test_index = test_idx[fold]

# X_train, X_test = X[train_index], X[test_index]
# y_train, y_test = y[train_index], y[test_index]

# regr.fit(X_train, y_train)

# y_pred = regr.predict(X_test)
# mae = metrics.mean_absolute_error(y_test, y_pred)
# print(mae)

# 487.935288456

kf = KFold(n_splits=10, random_state=np.random.RandomState(630), shuffle=True)
res = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    regr.fit(X_train,y_train)

    y_pred = regr.predict(X_test)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    res.append(mae)
    print('Mean Absolute Error:', mae)

    print('##############################################')
    # print(regr.feature_importances_)
    print(regr.oob_score_)
print('------------------------------')
print(res)
print(np.mean(res), np.std(res))

# 490.791550488 73.6824072046

# [0.00234327 0.00079034 0.01684722 0.25528921 0.06205532 0.40053958 0.13411685 0.03133449 0.04688988 0.04979384]