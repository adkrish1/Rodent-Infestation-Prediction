import mlflow.sklearn
import mlflow
import os


username = "your-access-key"
password = "your-secret-key"
uri = "129.114.26.209:8000/"

mlflow_username = os.getenv("MLFLOW_USERNAME")
mlflow_password = os.getenv("MLFLOW_PASSWORD")
mlflow_host = os.getenv("MLFLOW_HOST")
mlflow_port = os.getenv("MLFLOW_PORT")



mlflow.set_tracking_uri(
    f"http://{mlflow_username}:{mlflow_password}@{mlflow_host}:{mlflow_port}"
)

import pandas as pd
import numpy as np
import requests

import warnings
warnings.filterwarnings("ignore")

health_data =  pd.read_json('/mnt/rodent/data/43nn-pn8j - NYC Restaurant Inspection Results.json')

camis = health_data[['camis', 'boro']].drop_duplicates()


boro = 'Brooklyn'

health_data_agg = pd.read_csv("/mnt/rodent/data/restaurants_rodent.csv")

health_data_agg = health_data_agg.merge(camis, on = 'camis')
health_data_agg = health_data_agg[health_data_agg.boro == boro]

health_data_agg['inspection_month'] = pd.to_datetime(health_data_agg['inspection_month'])

health_data_agg.drop(columns = ['lat_rad', 'lon_rad', 'latitude', 'longitude', 'Unnamed: 0', 'boro'], inplace = True)


violation_cols = ['02G', '10F', '04L', '08A', '08C',
       '06C', '06E', '06D', '02H', '06F', '04M', '10B', '10G', '10I', '02B',
       '04H', '10H', '06A', '04K', '04N', '09B', '10E', '04J', '09C', '09A',
       '09E', '08B', '04A', '10J', '02C', '06B', '05D', '02I', '10A', '04C',
       '28-06', '06G', '10D', '03A', '05A', '03B', '10C', '05F', '28-05',
       '05B', '02A', '03E', '04O', '02F', '04D', '04E', '05H', '04P', '05C',
       '22G', '02D', '04F', '05E', '04I', '20-04', '03C', '03I', '16-03',
       '06H', '06I', '09D', '03G', '03F', '28-01', '28-07', '22F', '04B',
       '18-11', '03D', '20-06', '07A']

for lag in range(1, 4):
    health_data_agg[f'inspection_month_lag{lag}'] = (
        health_data_agg
          .groupby('camis')['inspection_month']
          .shift(lag)
    )


    health_data_agg[f'days_since_inspection_lag{lag}'] = (
        health_data_agg['inspection_month']
        - health_data_agg[f'inspection_month_lag{lag}']
    ).dt.days




    health_data_agg[f'score_lag{lag}'] = (
        health_data_agg.groupby('camis')['score']
          .shift(lag)
    )

    health_data_agg[f'rat_complaint_count{lag}'] = (
        health_data_agg.groupby('camis')['rat_complaint_count']
          .shift(lag)
    )


    for vc in violation_cols:
          health_data_agg[f'{vc}_lag{lag}'] = (
              health_data_agg
                .groupby('camis')[vc]
                .shift(lag)
          )

health_data_agg.drop(columns = ['inspection_month_lag1', 'inspection_month_lag2', 'inspection_month_lag3'], inplace = True)

health_data_agg['pred'] = 0
health_data_agg.loc[(health_data_agg['04L'] > 0) | (health_data_agg['04K'] > 0), 'pred'] = 1
health_data_agg.drop(columns = violation_cols, inplace = True)

health_data_agg.drop(columns = 'score', inplace = True)

"""***XGBoost***"""

mlflow.set_tracking_uri(
    f"http://{username}:{password}@{uri}"
)
mlflow.set_experiment(f"restaurant_infestation_predictor_test_" + boro)

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from sklearn import metrics

keys = health_data_agg['camis']
X = health_data_agg.drop(['camis','inspection_month',  'pred'], axis=1)
y = health_data_agg['pred']

X_train, X_test, y_train, y_test, keys_train, keys_test = train_test_split(
    X, y, keys,
    test_size=0.2,
    stratify=y,
    random_state=42
)

df_test_features = X_test.assign(key=keys_test)
df_test_features = df_test_features.assign(y = y_test)

df_test_features.to_csv(f"test_feats_" + boro + ".csv", index = False)

objective = 'auc'
n_estimators=200
learning_rate=0.05
max_depth=6
subsample=0.8
colsample_bytree=0.8
early_stopping_rounds=10

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
scale_pos_weight = neg / pos

model = xgb.XGBClassifier(
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric=objective,
    scale_pos_weight=scale_pos_weight,
    n_estimators=n_estimators,
    learning_rate=learning_rate,
    max_depth=max_depth,
    subsample=subsample,
    colsample_bytree=colsample_bytree,
    random_state=42,
    early_stopping_rounds=early_stopping_rounds,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=True
)

y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]


print("ROC AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))

roc_auc = metrics.roc_auc_score(y_test, y_proba)

# Accuracy
accuracy = metrics.accuracy_score(y_test, y_pred)

# F1 Score
f1 = metrics.f1_score(y_test, y_pred)

# Recall
recall = metrics.recall_score(y_test, y_pred)

# Precision
precision = metrics.precision_score(y_test, y_pred)

df_results = X_test.copy()
df_results['key']       = keys_test
df_results['actual']    = y_test
df_results['predicted'] = y_pred
df_results['proba_1']   = y_proba

import numpy as np

from sklearn.calibration import calibration_curve
import pandas as pd

bins = np.arange(0, 1.1, 0.1)

fraction_of_positives, mean_predicted_value = calibration_curve(df_results['actual'], df_results['proba_1'],
                                                                n_bins=10, strategy='uniform')


df_results['binned'] = pd.cut(df_results['proba_1'], bins=bins, include_lowest=True)


df_binned = df_results.groupby('binned').size().reset_index(name='count')

agg_df = df_results.groupby('binned').agg(
    actual_sum=('actual', 'sum'),
    actual_count=('actual', 'count') 
).reset_index()

agg_df['ratio'] = agg_df.actual_sum / agg_df.actual_count 
agg_df['perfect'] = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

agg_df.dropna(inplace = True)

x = np.arange(len(agg_df))
y = agg_df['ratio'].values  

slope, intercept = np.polyfit(x, y, 1) 

print(f"Slope of the ratio column: {slope}")

with mlflow.start_run():
    mlflow.log_param('eval_metric', 'auc')
    mlflow.log_param('scale_pos_weight', scale_pos_weight)
    mlflow.log_param('n_estimators', n_estimators)
    mlflow.log_param('learning_rate', learning_rate)
    mlflow.log_param('max_depth', max_depth)
    mlflow.log_param('subsample', subsample)
    mlflow.log_param('colsample_bytree', colsample_bytree)
    mlflow.log_param('early_stopping_rounds', early_stopping_rounds)
    mlflow.log_param('model_type', 'XGBoost')


    mlflow.log_metric('ROC AUC', roc_auc)
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('f1', f1)
    mlflow.log_metric('recall', recall)
    mlflow.log_metric('precision', precision)
    mlflow.log_metric('Calibration slope', slope)

    mlflow.sklearn.log_model(model, 'model')

import joblib

joblib.dump(model, 'xgb_model.joblib')
