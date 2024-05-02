# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time

# 초기전압설정 
initial_voltages = {
    'VOLT 1': 1.04, 'VOLT 2': 1.025, 'VOLT 3': 1.025, 'VOLT 4': 1.025788,
    'VOLT 5': 0.995631, 'VOLT 6': 1.012654, 'VOLT 7': 1.025769,
    'VOLT 8': 1.015883, 'VOLT 9': 1.032353
}

# 전압 데이터 생성 함수
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

# 데이터 전처리
def preprocess_data(df):
    features = df.drop('Time(s)', axis=1)
    return torch.tensor(features.values, dtype=torch.float32)

# 데이터셋 클래스
class VoltageDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 트랜스포머 모델
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

# 모델 훈련 함수
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

# 데이터 생성
data = generate_voltage_data(10, 0.5)

# 데이터 전처리
processed_data = preprocess_data(data)

# 데이터셋과 데이터 로더 준비
dataset = VoltageDataset(processed_data)
data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

# 모델 인스턴스 생성
model = TransformerModel(input_dim=9, output_dim=9, nhead=3, num_encoder_layers=3, dim_feedforward=512)

# 모델 훈련
train_model(model, data_loader)

import matplotlib.pyplot as plt

# 테스트 데이터 생성 함수 (실제 사용 시에는 실제 데이터를 로드)
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

# 모델 평가 및 결과 확인 함수
def evaluate_model(model, data_loader):
    model.eval()  # 모델을 평가 모드로 설정
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs in data_loader:
            outputs = model(inputs.unsqueeze(0))
            predictions.append(outputs.squeeze(0).numpy())
            actuals.append(inputs.numpy())

    return predictions, actuals

# 테스트 데이터 생성
test_data = generate_test_data(10, 0.5)

# 테스트 데이터 전처리
processed_test_data = preprocess_data(test_data)

# 테스트 데이터셋과 데이터 로더 준비
test_dataset = VoltageDataset(processed_test_data)
test_data_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 평가 모델
predictions, actuals = evaluate_model(model, test_data_loader)

# 결과 출력
for i, (pred, actual) in enumerate(zip(predictions, actuals)):
    if i < 5:  # 처음 5개 결과만 출력
        print(f"Prediction: {pred}, Actual: {actual}")

# 결과 시각화 (예측 vs 실제)
plt.figure(figsize=(15, 5))
plt.plot(np.array(actuals).flatten(), label='Actual Voltage')
plt.plot(np.array(predictions).flatten(), label='Predicted Voltage', linestyle='--')
plt.title('Voltage Prediction vs Actual')
plt.xlabel('Time Steps')
plt.ylabel('Voltage')
plt.legend()
plt.show()

