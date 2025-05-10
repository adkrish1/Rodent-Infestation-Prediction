import os
import pandas as pd
import numpy as np
import requests
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
import lightning as L
from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, Callback
from lightning.pytorch.loggers import MLFlowLogger
import mlflow
import mlflow.pytorch
import ray.train.lightning
from ray.train import ScalingConfig, RunConfig, FailureConfig
from ray.train.torch import TorchTrainer
from ray.train.lightning import RayTrainReportCallback

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback


mlflow_username = os.getenv("MLFLOW_USERNAME")
mlflow_password = os.getenv("MLFLOW_PASSWORD")
mlflow_host = os.getenv("MLFLOW_HOST")
mlflow_port = os.getenv("MLFLOW_PORT")
mlflow_uri = f"http://{mlflow_username}:{mlflow_password}@{mlflow_host}:{mlflow_port}"



class MLflowMetricsCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("train_loss")
        if loss is not None:
            mlflow.log_metric("train_loss", loss.item(), step=trainer.current_epoch)
    def on_test_end(self, trainer, pl_module):
        loss = trainer.callback_metrics.get("test_loss")
        if loss is not None:
            mlflow.log_metric("test_loss", loss.item(), step=trainer.current_epoch)
        mlflow.pytorch.log_model(pl_module, "model")


def get_edge_index(zipcodes_list):
    num_nodes = len(zipcodes_list)
    edges = []
    for i in range(num_nodes):
        if i > 2:
            edges.extend([[i, i-1], [i, i-2], [i, i-3]])
        if i < num_nodes - 3:
            edges.extend([[i, i+1], [i, i+2], [i, i+3]])
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def prepare_data(window_size):
    url = "https://data.cityofnewyork.us/resource/3q43-55fe.json"
    params = {"$limit": 300000, "complaint_type": "Rodent", "city": "BROOKLYN"}
    data = requests.get(url, params=params).json()
    complaints = pd.DataFrame(data)
    complaints["created_date"] = pd.to_datetime(complaints["created_date"])
    complaints["zipcode"] = complaints["incident_zip"].astype(str)
    complaints["month"] = complaints["created_date"].dt.to_period("M")
    rodent_counts = complaints.groupby(["zipcode","month"]).size().reset_index(name="rodent_complaints")

    url_h = "https://data.cityofnewyork.us/resource/43nn-pn8j.json"
    params_h = {"$limit": 300000, "boro": "Brooklyn"}
    hd = pd.DataFrame(requests.get(url_h, params=params_h).json())
    hd["score"] = pd.to_numeric(hd["score"], errors="coerce")
    hd["zipcode"] = hd["zipcode"].astype(str)
    hd["inspection_date"] = pd.to_datetime(hd["grade_date"])
    hd["month"] = hd["inspection_date"].dt.to_period("M")
    avg_health = hd.groupby(["zipcode","month"])["score"].mean().reset_index(name="avg_health_score")

    df = pd.merge(rodent_counts, avg_health, on=["zipcode","month"], how="left")
    df.drop(columns=["avg_health_score"], inplace=True)
    df = df[df["zipcode"]!="N/A"].sort_values(["zipcode","month"])
    zipcodes = sorted(df["zipcode"].unique())
    months = sorted(df["month"].unique())
    idx = pd.MultiIndex.from_product([zipcodes, months], names=["zipcode","month"])
    df = df.set_index(["zipcode","month"]).reindex(idx, fill_value=0).reset_index()

    data_dict = {(r["zipcode"], r["month"]): r["rodent_complaints"] for _,r in df.iterrows()}
    train_months = months[:-9]
    test_months = months[-9:]

    def make_samples(month_list):
        samples=[]
        for t in range(len(month_list)-1):
            if t < window_size: continue
            xs, ys, zcs = [], [], []
            for z in zipcodes:
                feats = [data_dict.get((z, month_list[p]), None) for p in range(t-window_size, t)]
                if None in feats: continue
                label = data_dict.get((z, month_list[t]), None)
                if label is None: continue
                xs.append(feats); ys.append(label); zcs.append(z)
            if xs:
                samples.append({"x":torch.tensor(xs,dtype=torch.float),
                                "y":torch.tensor(ys,dtype=torch.float),
                                "zipcodes":zcs})
        return samples

    return make_samples(train_months), make_samples(test_months)

class LightningGAT(L.LightningModule):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, lr):
        super().__init__()
        from torch_geometric.nn import GATConv
        self.save_hyperparameters()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels*heads, out_channels, heads=1, concat=False)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index); x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x.squeeze()

    def _common_step(self, batch, stage):
        y_hat = self(batch)
        loss = F.mse_loss(y_hat, batch.y)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._common_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


def tune_train(config):
    mlf_logger = MLFlowLogger(
        experiment_name="Ray_Test",
        tracking_uri=mlflow_uri
    )
    mlf_logger.log_hyperparams(config)

    train_samples, val_samples = prepare_data(config["window_size"])
    train_data = [
        Data(x=s["x"], edge_index=get_edge_index(s["zipcodes"]), y=s["y"])
        for s in train_samples
    ]
    val_data = [
        Data(x=s["x"], edge_index=get_edge_index(s["zipcodes"]), y=s["y"])
        for s in val_samples
    ]
    train_loader = GeoDataLoader(train_data, batch_size=config["batch_size"], shuffle=True)
    val_loader   = GeoDataLoader(val_data,   batch_size=config["batch_size"])

    model = LightningGAT(
        in_channels = config["window_size"],
        hidden_channels = config["hidden_channels"],
        out_channels = 1,
        heads = config["heads"],
        lr = config["lr"]
    )

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, mode="min"),
        ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min"),
        TuneReportCallback(
            {"val_loss":"val_loss","train_loss":"train_loss"},
            on="validation_end"
        )
    ]

    trainer = Trainer(
        max_epochs = config["num_epochs"],
        accelerator = "auto",
        devices     = "auto",
        logger      = mlf_logger,
        callbacks   = callbacks,
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, val_loader)

    best_path = trainer.checkpoint_callback.best_model_path
    best_model = LightningGAT.load_from_checkpoint(best_path)
    mlflow.pytorch.log_model(
        pytorch_model=best_model,
        artifact_path="best_lightning_model"
    )


search_space = {
    "window_size":      tune.choice([3,4,5]),
    "hidden_channels":  tune.choice([2,4,8,12]),
    "heads":            tune.choice([2,4,8,16]),
    "lr":               tune.loguniform(1e-3, 1e-1),
    "batch_size":       tune.choice([32,64]),
    "num_epochs":       50, 
}

scheduler = ASHAScheduler(
    metric="val_loss", mode="min",
    grace_period=5, reduction_factor=2
)
reporter = CLIReporter(
    parameter_columns=["window_size","hidden_channels","heads","lr","batch_size"],
    metric_columns=["val_loss","train_loss","training_iteration"]
)

if __name__ == "__main__":
    analysis = tune.run(
        tune_train,
        resources_per_trial={"cpu": 4, "gpu": 1},
        config=search_space,
        num_samples=20,
        scheduler=scheduler,
        progress_reporter=reporter,
        storage_path="s3://ray"
    )

    print("Best hyperparameters:", analysis.get_best_config("val_loss", "min"))
    print("Best validation loss:", analysis.best_result["val_loss"])