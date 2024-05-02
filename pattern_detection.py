import pandas as pd
import numpy as np
import time
import random

# 초기 전압, 변동 범위 및 회복 동작 정의
initial_voltages = {
    'VOLT 1': 1.04, 'VOLT 2': 1.025, 'VOLT 3': 1.025, 'VOLT 4': 1.025788,
    'VOLT 5': 0.995631, 'VOLT 6': 1.012654, 'VOLT 7': 1.025769,
    'VOLT 8': 1.015883, 'VOLT 9': 1.032353
}

fluctuation_range = 0.005
voltage_drop = 0.02  # 전압 강하의 크기
recovery_time = 1.0  # 회복이 시작되는 시간
drop_duration = 0.2  # 전압 강하 지속 시간

# 고장으로 간주되는 전압 임계값 계산
voltage_values = list(initial_voltages.values())
voltage_values.remove(recovery_time)
voltage_values.remove(failure_threshold)
average_voltage = sum(voltage_values) / (len(voltage_values))
failure_threshold = average_voltage * 0.85  # 평균 전압의 85% 이하로 떨어지면 고장으로 판단

# 전압 강하 패턴 저장
drop_pattern = []

# 변동 전압으로 새로운 데이터 행을 생성하는 함수
def generate_data(timestamp, start_time, drops_record, fault_record, volt, fault_times):
    new_data = {'Time(s)': timestamp - start_time}
    elapsed_time = timestamp - start_time

    fault_voltages = []

    if recovery_time < elapsed_time < recovery_time + drop_duration:
        drop_factor = 1 - voltage_drop  # 전압 강하
        new_data['Drop Occurred'] = 1  # 전압 강하 발생 표시
        drop_pattern.append(new_data)  # 전압 강하 패턴 저장
    else:
        drop_factor = 1  # 일반 변동
        new_data['Drop Occurred'] = 0  # 전압 강하 없음

    for v, init_value in initial_voltages.items():
        adjusted_voltage = np.random.normal(init_value * drop_factor, fluctuation_range)
        new_data[v] = adjusted_voltage

        # 고장 조건 검사
        if adjusted_voltage < failure_threshold:
            fault_voltages.append(v)
            if v == volt:
                fault_times[v] = new_data['Time(s)']  # 고장 발생 시간 기록

    # 전압 강하 이벤트 기록
    if new_data['Drop Occurred'] == 1:
        drops_record.append(new_data['Time(s)'])

    return new_data

# 실시간 데이터와 패턴 비교 및 고장 기록
def check_pattern(input_data, fault_record):
    if input_data == drop_pattern:
        print("실시간 데이터가 전압 강하 패턴과 일치합니다. 고장으로 기록합니다.")
        fault_record.update(fault_times)

# 데이터 생성 지속 시간 설정 (예: 30초)
duration = 30
start_time = time.time()
data = []
drops_record = []  # 전압 강하 발생 시간 기록
fault_record = {}  # 시간별 고장 기록
fault_times = {}  # 전압별 고장 발생 시간 기록

# 지정된 시간 동안 데이터 생성
while time.time() - start_time < duration:
    for volt in initial_voltages.keys():
        current_time = time.time()
        new_row = generate_data(current_time, start_time, drops_record, fault_record, volt, fault_times)
        data.append(new_row)

        # 실시간 데이터와 패턴 비교
        check_pattern(new_row, fault_record)

    time.sleep(0.5)  # 0.5초 간격으로 데이터 생성

# DataFrame으로 변환하여 시각화 또는 추가 분석 용이하게 함
generated_data = pd.DataFrame(data)

# 생성된 데이터, 전압 강하 발생 시간 및 고장 기록 출력
print(generated_data.head())
print("전압 강하 발생 시간:", drops_record)
print("기록된 고장:", fault_record)