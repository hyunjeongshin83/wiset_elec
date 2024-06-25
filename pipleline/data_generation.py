import pandas as pd
import numpy as np

def generate_device_data(device_name, model_number, mean_usage_time, voltage_pattern, n_samples=100):
    np.random.seed(42)
    usage_times = np.random.exponential(mean_usage_time, n_samples)
    voltages = np.random.normal(voltage_pattern, 0.5, n_samples)
    data = pd.DataFrame({
        'device': [device_name] * n_samples,
        'model_number': [model_number] * n_samples,
        'usage_time': usage_times,
        'voltage': voltages
    })
    return data

def save_critical_regions(data, start_duration=10, critical_times=None):
    start_region = data.copy()
    start_region['time'] = start_duration

    critical_region = data.copy()
    critical_region['time'] = critical_region.apply(
        lambda x: critical_times[x['device']], axis=1)
    
    return start_region, critical_region

if __name__ == "__main__":
    devices = [
        ('fan', 'FAN123', 2*60, 220),
        ('kettle', 'KET456', 1, 230),
        ('pc', 'PC789', 5*60, 250),
        ('monitor', 'MON101', 5*60, 240),
        ('hairdryer', 'HD102', 10, 200)
    ]

    device_data = pd.concat([generate_device_data(*device) for device in devices], ignore_index=True)

    critical_times = {
        'fan': 2*60,
        'kettle': 1,
        'pc': 5*60,
        'monitor': 5*60,
        'hairdryer': 10
    }

    start_region, critical_region = save_critical_regions(device_data, critical_times=critical_times)

    device_mapping = {'fan': 0, 'kettle': 1, 'pc': 2, 'monitor': 3, 'hairdryer': 4}
    start_region['device'] = start_region['device'].apply(lambda x: device_mapping[x])
    critical_region['device'] = critical_region['device'].apply(lambda x: device_mapping[x])

    start_region.to_csv("start_region.csv", index=False)
    critical_region.to_csv("critical_region.csv", index=False)
    device_data.to_csv("device_data.csv", index=False)
    print('Data generation complete.')
