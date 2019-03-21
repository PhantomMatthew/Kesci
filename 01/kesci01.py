import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

train_data = pd.read_csv('train_set.csv')

test_data = pd.read_csv('test_set.csv')

print(train_data.dtypes)

train_data.describe()

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

for col in train_data.columns[train_data.dtypes == 'object']:
    label_encoder = LabelEncoder()
    label_encoder.fit(train_data[col])

    train_data[col] = label_encoder.transform(train_data[col])
    test_data[col] = label_encoder.transform(test_data[col])

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)

scaler.fit(train_data[['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']])

train_data[['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']] = scaler.transform(
    train_data[['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']])
test_data[['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']] = scaler.transform(
    test_data[['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']])

from sklearn.model_selection import train_test_split

prediction = train_data['y']

train_use_col = list(set(train_data.columns) - set(['ID', 'y']))

features = train_data[train_use_col]

X_train, X_test, y_train, y_test = train_test_split(features, prediction, test_size=0.2, random_state=55)

print(features)

print("Training and testing split was successful.")

lgb_train = lgb.Dataset(X_train, label=y_train)
lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

# params = {
#     'boosting_type': 'gbdt',
#     'objective': 'binary',
#     'metric': {'auc'},
#     'num_leaves': 5,
#     'max_depth': 6,
#     'min_data_in_leaf': 450,
#     'learning_rate': 0.1,
#     'feature_fraction': 0.9,
#     'bagging_fraction': 0.95,
#     'bagging_freq': 5,
#     'lambda_l1': 1,
#     'lambda_l2': 0.001,
#     'min_gain_to_split': 0.2,
#     'verbose': 5,
#     'is_unbalance': True
# }

params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'auc'},
    # 'max_bin': 255,
    # 'num_leaves': 30,
    # 'max_depth': 5,
    # 'min_data_in_leaf': 450,
    'learning_rate': 0.02,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.90,
    'bagging_freq': 15,
    'lambda_l1': 2,
    'lambda_l2': 0.001,
    # 'min_gain_to_split': 0.2,
    # 'num_iterations': 100,
    'verbose': 0,
    'is_unbalance': True,
    # 'num_leaves' : 50,
    # 'num_trees' : 1000,
    # 'num_threads' : 32,
    # 'min_data_in_leaf' : 0,
    # 'min_sum_hessian_in_leaf' : 15
}

# gbm = lgb.train(params,
#                 lgb_train,
#                 num_boost_round=1000,
#                 valid_sets=lgb_eval,
#                 valid_names=None,
#                 fobj=None, feval=None, init_model=None,
#                 feature_name='auto', categorical_feature=['job', 'marital','education','default','housing','loan','contact','poutcome'],
#                 early_stopping_rounds=10, evals_result=None,
#                 verbose_eval=True,
#                 keep_training_booster=False, callbacks=None)

model = lgb.train(params,
                lgb_train,
                num_boost_round=10000,
                valid_sets=lgb_eval,
                feature_name='auto',
                # categorical_feature=['month','campaign','job', 'marital','education','default','contact','housing','contact','poutcome'],
                # categorical_feature=['campaign', 'job', 'marital', 'education', 'default', 'contact','housing', 'contact', 'poutcome'],
                # categorical_feature=['month','campaign','job', 'marital','education','contact','contact','poutcome'],
                # categorical_feature=['month','campaign','job', 'balance', 'marital','education','contact','contact','poutcome'],
                categorical_feature = 'auto',
                # categorical_feature=['duration','day','balance', 'balance', 'month','age','pdays','job','campaign', 'education'],
                early_stopping_rounds=1000
                )

preds = model.predict(test_data[train_use_col], num_iteration=model.best_iteration)

test_data['pred'] = preds

# submit_frame = pd.read_csv("submit.csv")

# result = pd.merge(submit_frame, test_raw, on='ID', how='left')
test_data[['ID', 'pred']].to_csv("result.csv", index=False)

plt.figure(figsize=(12,6))
lgb.plot_importance(model, max_num_features=30)
plt.title("Feature Importances")
plt.show()


# lgb_train = lgb.Dataset(X_train, label=y_train)
# lgb_eval = lgb.Dataset(X_test, label=y_test, reference=lgb_train)
#
# # params = {
# #     'boosting_type': 'gbdt',
# #     'objective': 'binary',
# #     'metric': {'auc'},
# #     'num_leaves': 5,
# #     'max_depth': 6,
# #     'min_data_in_leaf': 450,
# #     'learning_rate': 0.1,
# #     'feature_fraction': 0.9,
# #     'bagging_fraction': 0.95,
# #     'bagging_freq': 5,
# #     'lambda_l1': 1,
# #     'lambda_l2': 0.001,
# #     'min_gain_to_split': 0.2,
# #     'verbose': 5,
# #     'is_unbalance': True
# # }
#
# params = {
#     'task': 'train',
#     'boosting_type': 'gbdt',
#     'objective': 'binary',
#     'metric': {'auc'},
#     # 'max_bin': 255,
#     # 'num_leaves': 30,
#     # 'max_depth': 5,
#     # 'min_data_in_leaf': 450,
#     'learning_rate': 0.02,
#     'feature_fraction': 0.9,
#     'bagging_fraction': 0.90,
#     'bagging_freq': 15,
#     'lambda_l1': 2,
#     'lambda_l2': 0.001,
#     # 'min_gain_to_split': 0.2,
#     # 'num_iterations': 100,
#     'verbose': 0,
#     'is_unbalance': True,
#     # 'num_leaves' : 50,
#     # 'num_trees' : 1000,
#     # 'num_threads' : 32,
#     # 'min_data_in_leaf' : 0,
#     # 'min_sum_hessian_in_leaf' : 15
# }
#
# # gbm = lgb.train(params,
# #                 lgb_train,
# #                 num_boost_round=1000,
# #                 valid_sets=lgb_eval,
# #                 valid_names=None,
# #                 fobj=None, feval=None, init_model=None,
# #                 feature_name='auto', categorical_feature=['job', 'marital','education','default','housing','loan','contact','poutcome'],
# #                 early_stopping_rounds=10, evals_result=None,
# #                 verbose_eval=True,
# #                 keep_training_booster=False, callbacks=None)
#
# model = lgb.train(params,
#                 lgb_train,
#                 num_boost_round=10000,
#                 valid_sets=lgb_eval,
#                 feature_name='auto',
#                 categorical_feature=['month','campaign','job', 'marital','education','default','contact','housing','contact','poutcome'],
#                 # categorical_feature=['campaign', 'job', 'marital', 'education', 'default', 'contact','housing', 'contact', 'poutcome'],
#                 # categorical_feature=['month','campaign','job', 'marital','education','contact','contact','poutcome'],
#                 # categorical_feature=['month','campaign','job', 'balance', 'marital','education','contact','contact','poutcome'],
#                 # categorical_feature = 'auto',
#                 # categorical_feature=['duration','day','balance', 'balance', 'month','age','pdays','job','campaign', 'education'],
#                 early_stopping_rounds=1000
#                 )
# preds = model.predict(test_raw[train_use_col], num_iteration=model.best_iteration)
#
# test_raw['pred'] = preds
#
# # submit_frame = pd.read_csv("submit.csv")
#
# # result = pd.merge(submit_frame, test_raw, on='ID', how='left')
# test_raw[['ID', 'pred']].to_csv("result.csv",index=False)
#
# plt.figure(figsize=(12,6))
# lgb.plot_importance(model, max_num_features=30)
# plt.title("Feature Importances")
# plt.show()
