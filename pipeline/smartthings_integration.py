import pandas as pd
import requests
from keras.models import load_model
import numpy as np

#SMARTTHINGS_API_URL = 'https://api.smartthings.com/v1/devices'
#SMARTTHINGS_API_TOKEN = 'YOUR_SMARTTHINGS_API_TOKEN'

# 임시 데이터 (테스트용)
start_region = pd.DataFrame({
    'time': [1, 2, 3],
    'usage_time': [3.5, 2.0, 5.5],
    'voltage': [220, 240, 210],
    'model_number': ['model1', 'model2', 'model3']
})

critical_region = pd.DataFrame({
    'time': [4, 5, 6],
    'usage_time': [6.0, 7.5, 8.2],
    'voltage': [250, 230, 200],
    'model_number': ['model4', 'model5', 'model6']
})

# 모델 로드 (실제 모델 대신 임시로 사용)
def load_dummy_model():
    return np.random.rand(1)

# 평균 사용 시간 (실제 데이터 대신 임시로 사용)
mean_usage_time = {
    'fan': 2*60,
    'kettle': 1,
    'pc': 5*60,
    'monitor': 5*60,
    'hairdryer': 10
}

# 장치 매핑 (실제 데이터 대신 임시로 사용)
device_mapping = {'fan': 0, 'kettle': 1, 'pc': 2, 'monitor': 3, 'hairdryer': 4}

# 등록된 장치 (실제 데이터 대신 임시로 사용)
registered_devices = ['model1', 'model3', 'model4']

# SmartThings API 호출 시 연결 오류 방지를 위한 임시 함수
def mock_notify_smartthings(device_id, message):
    print(f"Mock Notification sent: {message}")

# 비정상 탐지 및 알림 함수 (실제 데이터 대신 임시로 사용)
def detect_anomalies_and_notify(data, model, mean_usage_time, registered_devices):
    anomalies = []
    # 임시 예측
    predictions = np.random.rand(len(data))
    for index, row in data.iterrows():
        device_type = int(predictions[index] > 0.5)
        device_name = list(device_mapping.keys())[list(device_mapping.values()).index(device_type)]
        if device_name in mean_usage_time and row['usage_time'] > mean_usage_time[device_name]:
            anomalies.append((index, device_name, row['usage_time'], row['model_number']))
            if row['model_number'] in registered_devices:
                message = f"Fire risk warning: Device={device_name}, Usage Time={row['usage_time']:.2f} minutes, Model Number={row['model_number']}"
                mock_notify_smartthings(row['model_number'], message)  # 실제 notify_smartthings 대신 mock_notify_smartthings 사용
    return anomalies

if __name__ == "__main__":
    # 임시 데이터로 detect_anomalies_and_notify 함수 호출
    anomalies_start = detect_anomalies_and_notify(start_region, load_dummy_model(), mean_usage_time, registered_devices)
    anomalies_critical = detect_anomalies_and_notify(critical_region, load_dummy_model(), mean_usage_time, registered_devices)

    print('SmartThings integration complete.')
