

import pandas as pd
import numpy as np
import requests
import torch
from torch_geometric.data import Data

from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import GATConv
import torch.nn.functional as F
import mlflow
import mlflow.pytorch


username = os.getenv("MLFLOW_USERNAME")
password = os.getenv("MLFLOW_PASSWORD")
host = os.getenv("MLFLOW_HOST")
port = os.getenv("MLFLOW_PORT")

tracking_uri = f"http://{username}:{password}@{host}:{port}"

mlflow.set_tracking_uri(tracking_uri)

mlflow.set_experiment('Ray_Test')

import requests

url = "https://data.cityofnewyork.us/resource/3q43-55fe.json"

params = {
    "$limit": 300000,  # Number of records
    "complaint_type": "Rodent",
    "city": "BROOKLYN"
}

response = requests.get(url, params=params)

if response.status_code == 200:
    data = response.json()

    print(f"Retrieved {len(data)} records.")
else:
    print(f"Failed to retrieve data: {response.status_code}")

complaints_data = pd.DataFrame(data)
complaints_data['created_date'] = pd.to_datetime(complaints_data['created_date'])
complaints_data['zipcode'] = complaints_data['incident_zip'].astype(str)
complaints_data['month'] = complaints_data['created_date'].dt.to_period('M')

rodent_counts = complaints_data.groupby(['zipcode', 'month']).size().reset_index(name='rodent_complaints')

rodent_counts

params = {
    "$limit": 300000,  # Number of records
    "boro" : "Brooklyn"
}

url_health = "https://data.cityofnewyork.us/resource/43nn-pn8j.json"
response_health = requests.get(url_health, params = params)
health_data = pd.DataFrame(response_health.json())

health_data['score'] = pd.to_numeric(health_data['score'], errors='coerce')
health_data['zipcode'] = health_data['zipcode'].astype(str)
health_data['inspection_date'] = pd.to_datetime(health_data['grade_date'])
health_data['month'] = health_data['inspection_date'].dt.to_period('M')
health_data['key'] = health_data['dba'] + health_data['building']

avg_health_score = health_data.groupby(['zipcode', 'month'])['score'].mean().reset_index()
avg_health_score.rename(columns={'score': 'avg_health_score'}, inplace=True)

df = pd.merge(rodent_counts, avg_health_score, on=["zipcode", "month"], how="left")

df.drop(columns = ['avg_health_score'], inplace = True)
df = df.sort_values(by=['zipcode', 'month'])

zipcodes = df['zipcode'].unique()
months = df['month'].unique()

all_combinations = pd.MultiIndex.from_product([zipcodes, months], names=['zipcode', 'month'])

df.set_index(['zipcode', 'month'], inplace=True)

df = df.reindex(all_combinations, fill_value=0)

df.reset_index(inplace=True)
df = df[df.zipcode != 'N/A']


window_size = 3


data_dict = {(row['zipcode'], row['month']): row['rodent_complaints'] for _, row in df.iterrows()}

unique_months = sorted(df['month'].unique())
zipcodes = sorted(df['zipcode'].unique())

train_months = unique_months[:-9] 
test_months = unique_months[-9:]

training_samples = []

for t in range(window_size, len(train_months) - 1):
    current_month = train_months[t]
    next_month = train_months[t+1]

    x_features = []
    y_targets = []
    available_zipcodes = []

    for zipcode in zipcodes:
        features = []
        valid = True

        for past_t in range(t - window_size, t):
            month_val = train_months[past_t]

            if (zipcode, month_val) in data_dict:
                features.append(data_dict[(zipcode, month_val)])
            else:
                valid = False
                break
        if valid and ((zipcode, next_month) in data_dict):
            x_features.append(features)
            y_targets.append(data_dict[(zipcode, next_month)])
            available_zipcodes.append(zipcode)

    if len(x_features) == 0:
        continue

    x_tensor = torch.tensor(x_features, dtype=torch.float)
    y_tensor = torch.tensor(y_targets, dtype=torch.float)

    sample = {
        'x': x_tensor,
        'y': y_tensor,
        'zipcodes': available_zipcodes,
        'current_time': current_month,
        'next_time': next_month
    }
    training_samples.append(sample)

print(f"Created {len(training_samples)} training samples.")

test_samples = []


for t in range(window_size, len(test_months) - 1):
    current_month = test_months[t]
    next_month = test_months[t+1]

    x_features = []
    y_targets = []
    available_zipcodes = []
    # print(zipcodes)
    for zipcode in zipcodes:
        features = []
        valid = True

        for past_t in range(t - window_size, t):
            month_val = test_months[past_t]
            # print(month_val)

            if (zipcode, month_val) in data_dict:
                features.append(data_dict[(zipcode, month_val)])
            else:
                valid = False
                break
        if valid and ((zipcode, next_month) in data_dict):
            x_features.append(features)
            y_targets.append(data_dict[(zipcode, next_month)])
            available_zipcodes.append(zipcode)

    if len(x_features) == 0:
        continue

    x_tensor = torch.tensor(x_features, dtype=torch.float)
    y_tensor = torch.tensor(y_targets, dtype=torch.float)

    sample = {
        'x': x_tensor,
        'y': y_tensor,
        'zipcodes': available_zipcodes,
        'current_time': current_month,
        'next_time': next_month
    }
    test_samples.append(sample)

print(f"Created {len(test_samples)} test samples.")

import torch
from torch_geometric.data import Data

def get_edge_index(zipcodes_list):

    num_nodes = len(zipcodes_list)
    edge_list = []

    for i in range(num_nodes):
        if i > 2:
            edge_list.append([i, i - 1])
            edge_list.append([i, i - 2])
            edge_list.append([i, i - 3])
        if i < num_nodes - 3:
            edge_list.append([i, i + 1])
            edge_list.append([i, i + 2])
            edge_list.append([i, i + 3])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index

sample = training_samples[0]
edge_index = get_edge_index(sample['zipcodes'])


data = Data(x=sample['x'], y=sample['y'], edge_index=edge_index)

print("Edge index tensor:")
print(edge_index)
print("\nGraph Data Object:")
print(data)

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4):

        super(GAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return x.squeeze()

hidden_channels = 4
out_channels = 1
heads = 2
lr = 0.01
num_epochs = 100



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = GAT(in_channels=window_size, hidden_channels=hidden_channels, out_channels=out_channels, heads=heads).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=lr)
num_epochs = num_epochs


with mlflow.start_run():
    mlflow.log_param("window_size", window_size)
    mlflow.log_param("hidden_channels", hidden_channels)
    mlflow.log_param("out_channels", out_channels)
    mlflow.log_param("heads", heads)
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("num_epochs", num_epochs)

  
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0.0
        for sample in training_samples:
        
            edge_index = get_edge_index(sample['zipcodes']).to(device)
            x = sample['x'].to(device)
            y = sample['y'].to(device)

            optimizer.zero_grad()

            out = model(x, edge_index)

            loss = F.mse_loss(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(training_samples)

        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)


        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for sample in test_samples:
                edge_index = get_edge_index(sample['zipcodes']).to(device)
                x = sample['x'].to(device)
                y = sample['y'].to(device)


                out = model(x, edge_index)

                test_loss = F.mse_loss(out, y)
                total_test_loss += test_loss.item()

        avg_test_loss = total_test_loss / len(test_samples)

        mlflow.log_metric("test_loss", avg_test_loss, step=epoch)


        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

    mlflow.pytorch.log_model(model, "model")

    print("Training finished and model logged to MLflow!")

print("Actual vs Predicted for all test samples:")

actual_vs_predicted = []

for sample in test_samples:
    edge_index = get_edge_index(sample['zipcodes']).to(device)
    x = sample['x'].to(device)
    y = sample['y'].to(device)


    model.eval()
    with torch.no_grad():
        predicted = model(x, edge_index)

    for i, zipcode in enumerate(sample['zipcodes']):
        actual_value = y[i].item()  
        predicted_value = predicted[i].item() 
        actual_vs_predicted.append((zipcode, actual_value, predicted_value))

for zipcode, actual, predicted in actual_vs_predicted:
    print(f"Zipcode: {zipcode}, Actual: {actual:.4f}, Predicted: {predicted:.4f}")