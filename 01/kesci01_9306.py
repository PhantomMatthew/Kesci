import lightgbm as lgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

train_raw = pd.read_csv('train_set.csv')

#data
test_raw = pd.read_csv('test_set.csv')

#test_data
print(train_raw.dtypes)

train_raw.describe()

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

for col in train_raw.columns[train_raw.dtypes == 'object']:
    le = LabelEncoder()
    le.fit(train_raw[col])

    train_raw[col] = le.transform(train_raw[col])
    test_raw[col] = le.transform(test_raw[col])

# print(train_raw)
# print(test_raw)

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)

scaler.fit(train_raw[['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']])

train_raw[['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']] = scaler.transform(
    train_raw[['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']])
test_raw[['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']] = scaler.transform(
    test_raw[['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']])

train_raw.head(100)
test_raw.head(100)

from sklearn.model_selection import train_test_split

prediction = train_raw['y']

train_use_col = list(set(train_raw.columns) - set(['ID', 'y']))

features = train_raw[train_use_col]

X_train, X_test, y_train, y_test = train_test_split(features, prediction, test_size=0.2, random_state=42)

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
    'learning_rate': 0.01,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.90,
    'bagging_freq': 12,
    'lambda_l1': 2,
    'lambda_l2': 0.002,
    # 'min_gain_to_split': 0.2,
    # 'num_iterations': 100,
    'verbose': 0,
    'is_unbalance': True
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
                # categorical_feature=['job', 'marital','education','default','housing','loan','contact','poutcome'],
                categorical_feature = 'auto',
                early_stopping_rounds=1000
                )
preds = model.predict(test_raw[train_use_col], num_iteration=model.best_iteration)

test_raw['pred'] = preds

# submit_frame = pd.read_csv("submit.csv")

# result = pd.merge(submit_frame, test_raw, on='ID', how='left')
test_raw[['ID', 'pred']].to_csv("result.csv",index=False)

plt.figure(figsize=(12,6))
lgb.plot_importance(model, max_num_features=30)
plt.title("Feature Importances")
plt.show()
