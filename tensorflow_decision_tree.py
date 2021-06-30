# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 13:31:59 2021

@author: zhou wu
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder


import gc
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import tensorflow_decision_forests as tfdf
tfdf.keras.get_all_models()
model = tfdf.keras.GradientBoostedTreesModel()
sample_submission = pd.read_csv('sample_submission.csv')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train.head()
test_features = test.columns[1:]
train_features = train.columns[1:]
target = train['target'].values
train_oof = np.zeros((train.shape[0],9))
test_preds = 0
test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test[test_features])
n_splits = 5
kf = KFold(n_splits=n_splits, random_state=137, shuffle=True)

for jj, (train_index, val_index) in enumerate(kf.split(train)):
    print("Fitting fold", jj+1)
    train_ds = train.loc[train_index]

    val_ds = train.loc[val_index]
    
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_ds[train_features], label="target")
    val_ds = tfdf.keras.pd_dataframe_to_tf_dataset(val_ds[train_features], label="target")
    model = tfdf.keras.GradientBoostedTreesModel(num_trees=700, subsample = 0.9, max_depth = 3)
    model.fit(x=train_ds)
    val_pred = model.predict(val_ds)
    train_oof[val_index] = val_pred
    #print("Fold AUC:", roc_auc_score(val_target, val_pred[:,1]))
    test_preds += model.predict(test_ds)/n_splits
    del train_ds, val_ds
    gc.collect()
oho = OneHotEncoder()
log_loss(oho.fit_transform(target.reshape(-1,1)).toarray(), train_oof)
sample_submission[sample_submission.columns[1:]] = test_preds
sample_submission.to_csv('submission.csv', index=False)