import serial
import time
import re
import csv
import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib
import matplotlib.pyplot as plt

# ====== 한글 폰트 설정 (윈도우 기준) ======
matplotlib.rcParams['font.family'] = 'Malgun Gothic'  # Windows
matplotlib.rcParams['axes.unicode_minus'] = False     # 마이너스 깨짐 방지
# ========================================

# ==== CSV 저장 ====
def save_experiment_data(times, total_time, file_path='experiment_data.csv'):
    header = ['t1', 't2', 't3', 't4', 't5', 'total']
    data_row = times + [total_time]
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(data_row)

# ==== 시리얼로부터 수위 시간 읽기 ====
def read_times_and_total_from_arduino(ser, max_wait_time=120):
    times = []
    total_time = None
    pattern_time = re.compile(r"✅ .* 시간\(ms\): (\d+)")
    pattern_total = re.compile(r"💧 물 가득 찼을 때 시간\(ms\): (\d+)")
    start_time = time.time()

    print("[📡 수신 시작] 아두이노로부터 시간 데이터 수신 중...")

    while True:
        if time.time() - start_time > max_wait_time:
            print("⏱ 최대 대기 시간 초과. 현재까지 받은 데이터로 예측 시도합니다.")
            break

        if ser.in_waiting:
            line = ser.readline().decode(errors='ignore').strip()
            print(f"[시리얼] {line}")

            match_time = pattern_time.search(line)
            if match_time and len(times) < 5:
                ms = int(match_time.group(1))
                sec = ms / 1000.0
                times.append(sec)
                print(f"✅ 받은 시간값 #{len(times)}: {sec:.2f}초")

            match_total = pattern_total.search(line)
            if match_total:
                ms_total = int(match_total.group(1))
                total_time = ms_total / 1000.0
                print(f"💧 총 소요 시간 받음: {total_time:.2f}초")

            if len(times) == 5 and total_time is not None:
                break

    print(f"[🎉 완료] 측정된 시간값: {times}, 총 시간: {total_time if total_time else '미수신'}")
    return times, total_time

# ==== LSTM 학습 + 예측 ====
def train_and_predict_lstm(csv_path='experiment_data.csv', reference_time=70.0, current_input=None):
    if not os.path.exists(csv_path):
        print("❌ 학습할 CSV 데이터가 없습니다.")
        return None, None

    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    X = data[:, :-1]
    y = data[:, -1]

    if len(X) < 3:
        print("⚠️ 학습 데이터가 부족합니다 (3회 이상 필요). 예측을 생략합니다.")
        return None, None

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    X_lstm = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

    model = Sequential()
    model.add(LSTM(64, input_shape=(X_lstm.shape[1], 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_lstm, y_scaled, epochs=300, verbose=0)

    if current_input is not None:
        padded = current_input + [0] * (5 - len(current_input))
        scaled_input = scaler_X.transform([padded])
        scaled_input = scaled_input.reshape((1, 5, 1))

        predicted_scaled = model.predict(scaled_input)[0]
        predicted_time = scaler_y.inverse_transform([predicted_scaled])[0][0]
        time_diff = predicted_time - reference_time

        print(f"\n📈 예측된 물이 다 찰 시간: {predicted_time:.2f}초")
        print(f"📏 기준 시간 (70초)과 차이: {time_diff:.2f}초")

        # 그래프 출력
        plt.figure(figsize=(8, 4))

        # x축 1부터 시작
        x_actual = np.arange(1, len(y) + 1)

        # 실제 총 시간 (파란 점)
        plt.plot(x_actual, y, 'bo-', label='실제 총 시간')

        # 마지막 실험 실제값 초록색 점
        plt.plot(x_actual[-1], y[-1], 'go', label='마지막 실험 실제값')

        # 예측값 위치: 마지막 실험 번호 + 0.5
        x_pred = x_actual[-1] + 0.5
        plt.plot(x_pred, predicted_time, 'ro', label='예측값 (현재 입력 기반)')

        # x축 레이블에 '예측값' 추가
        plt.xticks(list(x_actual) + [x_pred], list(map(str, x_actual)) + ['예측값'])

        plt.axhline(reference_time, color='k', linestyle='--', label='기준값 70초')
        plt.xlabel("실험 번호")
        plt.ylabel("총 소요 시간 (초)")
        plt.title("LSTM 기반 예측 결과")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        return predicted_time, time_diff
    else:
        print("⚠️ 예측을 위한 입력 데이터가 없습니다.")
        return None, None

# ==== 메인 함수 ====
def main():
    port = 'COM7'  # ⚠️ 시리얼 포트 확인 필수
    baudrate = 9600

    try:
        with serial.Serial(port, baudrate, timeout=10) as ser:
            time.sleep(2)  # 시리얼 초기화 대기
            times, total_time = read_times_and_total_from_arduino(ser)

            if len(times) == 5 and total_time is not None:
                save_experiment_data(times, total_time)
                print("[💾 저장 완료] CSV에 저장되었습니다.")
            else:
                print("⚠️ 데이터가 완전하지 않아 CSV 저장하지 않습니다.")

            if len(times) >= 1:
                train_and_predict_lstm(current_input=times)
            else:
                print("⛔ 예측을 위한 시간 데이터가 부족합니다.")
    except serial.SerialException:
        print("❌ 시리얼 포트를 열 수 없습니다. 포트 이름과 연결 상태를 확인하세요.")

if __name__ == "__main__":
    main()
