
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score,
    f1_score, recall_score, precision_score
)

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.air.integrations.mlflow import MLflowLoggerCallback
import mlflow


USERNAME = os.getenv("MLFLOW_USERNAME")
PASSWORD = os.getenv("MLFLOW_PASSWORD")
HOST = os.getenv("MLFLOW_HOST")
PORT = os.getenv("MLFLOW_PORT")

URI      = f"http://{USERNAME}:{PASSWORD}@{HOST}:{PORT}"
EXPERIMENT = "restaurant_infestation_predictor_test_Brooklyn"

import os
os.environ["MLFLOW_TRACKING_URI"] = URI
os.environ["MLFLOW_EXPERIMENT_NAME"] = EXPERIMENT


HEALTH_JSON = "/mnt/rodent/data/43nn-pn8j - NYC Restaurant Inspection Results.json"
RODENT_CSV  = "/mnt/rodent/data/restaurants_rodent.csv"
violation_cols = ['02G', '10F', '04L', '08A', '08C',
       '06C', '06E', '06D', '02H', '06F', '04M', '10B', '10G', '10I', '02B',
       '04H', '10H', '06A', '04K', '04N', '09B', '10E', '04J', '09C', '09A',
       '09E', '08B', '04A', '10J', '02C', '06B', '05D', '02I', '10A', '04C',
       '28-06', '06G', '10D', '03A', '05A', '03B', '10C', '05F', '28-05',
       '05B', '02A', '03E', '04O', '02F', '04D', '04E', '05H', '04P', '05C',
       '22G', '02D', '04F', '05E', '04I', '20-04', '03C', '03I', '16-03',
       '06H', '06I', '09D', '03G', '03F', '28-01', '28-07', '22F', '04B',
       '18-11', '03D', '20-06', '07A']

def load_and_prepare():
    health = pd.read_json(HEALTH_JSON)
    rodent = pd.read_csv(RODENT_CSV)
    camis  = health[['camis','boro']].drop_duplicates()
    df     = rodent.merge(camis, on='camis')
    df     = df[df.boro == "Brooklyn"].copy()
    df['inspection_month'] = pd.to_datetime(df['inspection_month'])
    df.drop(columns=['lat_rad','lon_rad','latitude','longitude','Unnamed: 0','boro'], inplace=True)

    for lag in (1,2,3):
        df[f'inspection_month_lag{lag}'] = df.groupby('camis')['inspection_month'].shift(lag)
        df[f'days_since_inspection_lag{lag}'] = (
            (df['inspection_month'] - df[f'inspection_month_lag{lag}']).dt.days
        )
        df[f'score_lag{lag}'] = df.groupby('camis')['score'].shift(lag)
        df[f'rat_complaint_count{lag}'] = df.groupby('camis')['rat_complaint_count'].shift(lag)
        for vc in violation_cols:
            df[f'{vc}_lag{lag}'] = df.groupby('camis')[vc].shift(lag)

    df['pred'] = 0
    df.loc[(df['04L']>0)|(df['04K']>0), 'pred'] = 1

    df.drop(
      columns=violation_cols
              + [f'inspection_month_lag{l}' for l in (1,2,3)]
              + ['score'],
      inplace=True
    )
    X = df.drop(['camis','inspection_month','pred'], axis=1)
    y = df['pred']
    return X, y

_X, _y = load_and_prepare()
imbalance_ratio = (_y==0).sum() / (_y==1).sum()


def train_xgb(config):
    X, y = load_and_prepare()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    config["scale_pos_weight"] = imbalance_ratio

    model = xgb.XGBClassifier(**config)
    model.fit(X_train, y_train, eval_set=[(X_test,y_test)], verbose=False)

    y_proba = model.predict_proba(X_test)[:,1]
    y_pred  = model.predict(X_test)
    metrics = {
        "roc_auc":   roc_auc_score(y_test, y_proba),
        "accuracy":  accuracy_score(y_test, y_pred),
        "f1":        f1_score(y_test, y_pred),
        "recall":    recall_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
    }
    tune.report(metrics)

search_space = {
    "objective":             "binary:logistic",
    "use_label_encoder":     False,
    "eval_metric":           "auc",
    "n_estimators":          tune.choice([100, 200, 300]),
    "learning_rate":         tune.loguniform(1e-3, 1e-1),
    "max_depth":             tune.choice([4, 6, 8]),
    "subsample":             tune.uniform(0.6, 1.0),
    "colsample_bytree":      tune.uniform(0.6, 1.0),
    "early_stopping_rounds": tune.choice([5, 10, 20]),
    "random_state":          42,
}

scheduler = ASHAScheduler(metric="roc_auc", mode="max", grace_period=5, reduction_factor=2)
reporter  = CLIReporter(
    parameter_columns=["n_estimators","max_depth","learning_rate"],
    metric_columns=["roc_auc","accuracy","training_iteration"]
)

analysis = tune.run(
    train_xgb,
    resources_per_trial={"cpu": 8, "gpu": 1},
    config=search_space,
    num_samples=10,
    scheduler=scheduler,
    progress_reporter=reporter,
    callbacks=[
      MLflowLoggerCallback(
        tracking_uri=URI,
        experiment_name=EXPERIMENT
      )
    ],
    storage_path="s3://ray"
)

print("Best hyperparameters:", analysis.get_best_config(metric="roc_auc", mode="max"))
best_config = analysis.get_best_config(metric="roc_auc", mode="max")
print("Best config:", best_config)


X, y = load_and_prepare()  

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

neg = (y_train==0).sum()
pos = (y_train==1).sum()
best_config["scale_pos_weight"] = neg/pos

best_model = xgb.XGBClassifier(**best_config)
best_model.fit(X_train, y_train, eval_set=[(X_val,y_val)], verbose=False)

mlflow.set_tracking_uri(f"http://{USERNAME}:{PASSWORD}@{HOST}:{PORT}")
mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run(run_name="best_xgb_model"):

    mlflow.log_params(best_config)

    y_proba = best_model.predict_proba(X_val)[:,1]
    y_pred  = best_model.predict(X_val)
    mlflow.log_metric("roc_auc",   roc_auc_score(y_val, y_proba))
    mlflow.log_metric("accuracy",  accuracy_score(y_val, y_pred))
    mlflow.log_metric("f1",        f1_score(y_val, y_pred))
    mlflow.log_metric("recall",    recall_score(y_val, y_pred))
    mlflow.log_metric("precision", precision_score(y_val, y_pred))

    mlflow.sklearn.log_model(best_model, "best_model")
