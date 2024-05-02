# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time

# �ʱ����м��� 
initial_voltages = {
    'VOLT 1': 1.04, 'VOLT 2': 1.025, 'VOLT 3': 1.025, 'VOLT 4': 1.025788,
    'VOLT 5': 0.995631, 'VOLT 6': 1.012654, 'VOLT 7': 1.025769,
    'VOLT 8': 1.015883, 'VOLT 9': 1.032353
}

# ���� ������ ���� �Լ�
def generate_voltage_data(duration, interval):
    data = []
    start_time = time.time()
    while time.time() - start_time < duration:
        current_time = time.time()
        voltages = {key: np.random.normal(val, 0.005) for key, val in initial_voltages.items()}
        voltages['Time(s)'] = current_time - start_time
        data.append(voltages)
        time.sleep(interval)
    return pd.DataFrame(data)

# ������ ��ó��
def preprocess_data(df):
    features = df.drop('Time(s)', axis=1)
    return torch.tensor(features.values, dtype=torch.float32)

# �����ͼ� Ŭ����
class VoltageDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# Ʈ�������� ��
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, nhead, num_encoder_layers, dim_feedforward):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(
            d_model=input_dim,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward
        )
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, src):
        output = self.transformer(src, src)
        return self.fc_out(output)

# �� �Ʒ� �Լ�
def train_model(model, data_loader, epochs=10):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        for inputs in data_loader:
            optimizer.zero_grad()
            outputs = model(inputs.unsqueeze(0))
            loss = criterion(outputs, inputs.unsqueeze(0))
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# ������ ����
data = generate_voltage_data(10, 0.5)

# ������ ��ó��
processed_data = preprocess_data(data)

# �����ͼ°� ������ �δ� �غ�
dataset = VoltageDataset(processed_data)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# �� �ν��Ͻ� ����
model = TransformerModel(input_dim=9, output_dim=9, nhead=3, num_encoder_layers=3, dim_feedforward=512)

# �� �Ʒ�
train_model(model, data_loader)

import matplotlib.pyplot as plt

# �׽�Ʈ ������ ���� �Լ� (���� ��� �ÿ��� ���� �����͸� �ε�)
def generate_test_data(duration, interval):
    data = []
    start_time = time.time()
    while time.time() - start_time < duration:
        current_time = time.time()
        voltages = {key: np.random.normal(val, 0.005) for key, val in initial_voltages.items()}
        voltages['Time(s)'] = current_time - start_time
        data.append(voltages)
        time.sleep(interval)
    return pd.DataFrame(data)

# �� �� �� ��� Ȯ�� �Լ�
def evaluate_model(model, data_loader):
    model.eval()  # ���� �� ���� ����
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs in data_loader:
            outputs = model(inputs.unsqueeze(0))
            predictions.append(outputs.squeeze(0).numpy())
            actuals.append(inputs.numpy())

    return predictions, actuals

# �׽�Ʈ ������ ����
test_data = generate_test_data(10, 0.5)

# �׽�Ʈ ������ ��ó��
processed_test_data = preprocess_data(test_data)

# �׽�Ʈ �����ͼ°� ������ �δ� �غ�
test_dataset = VoltageDataset(processed_test_data)
test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# �� ��
predictions, actuals = evaluate_model(model, test_data_loader)

# ��� ���
for i, (pred, actual) in enumerate(zip(predictions, actuals)):
    if i < 5:  # ó�� 5�� ����� ���
        print(f"Prediction: {pred}, Actual: {actual}")

# ��� �ð�ȭ (���� vs ����)
plt.figure(figsize=(15, 5))
plt.plot(np.array(actuals).flatten(), label='Actual Voltage')
plt.plot(np.array(predictions).flatten(), label='Predicted Voltage', linestyle='--')
plt.title('Voltage Prediction vs Actual')
plt.xlabel('Time Steps')
plt.ylabel('Voltage')
plt.legend()
plt.show()

