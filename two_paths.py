import numpy as np
import matplotlib.pyplot as plt
import json

# t 시간 설정
t = np.linspace(0, 10, 100)

# 1. 처음에 빠르게 가다가 나중에 천천히 (로그 함수 기반, 3가지 버전)
def fast_then_slow1(t):
    return 10 * (np.log(9*t/10 + 1) / np.log(10))

def fast_then_slow2(t):
    return 10 * (np.log(7*t/10 + 1) / np.log(8))

def fast_then_slow3(t):
    return 10 * (np.log(5*t/10 + 1) / np.log(6))

# 2. 처음에 느리게 갔다가 나중에 빠르게 (지수 함수 기반, 3가지 버전)
def slow_then_fast1(t):
    return 10 * (np.exp(0.23*t) - 1) / (np.exp(2.3) - 1)

def slow_then_fast2(t):
    return 10 * (np.exp(0.3*t) - 1) / (np.exp(3.0) - 1)

def slow_then_fast3(t):
    return 10 * (np.exp(0.35*t) - 1) / (np.exp(3.5) - 1)

# 3. 두 가지 다른 주기의 증가 함수
def periodic_speed1(t):
    return t + 0.5 * (1 - np.cos(2*np.pi*t/5))  # 5초 주기

def periodic_speed2(t):
    return t + 0.5 * (1 - np.cos(2*np.pi*t/2.5))  # 2.5초 주기

# 4. 직선 (균일한 속도)
def constant_speed(t):
    return t

# 모든 경로 함수들을 딕셔너리로 저장
path_functions = {
    'fast_then_slow1': fast_then_slow1,
    'fast_then_slow2': fast_then_slow2,
    'fast_then_slow3': fast_then_slow3,
    'slow_then_fast1': slow_then_fast1,
    'slow_then_fast2': slow_then_fast2,
    'slow_then_fast3': slow_then_fast3,
    'periodic_speed1': periodic_speed1,
    'periodic_speed2': periodic_speed2,
    'constant_speed': constant_speed
}

# 경로 데이터 생성 및 저장
def save_path_data(filename='path_data.json'):
    path_data = {}
    t_list = t.tolist()  # numpy array를 list로 변환
    
    for name, func in path_functions.items():
        path_data[name] = {
            't': t_list,
            'x': func(t).tolist()
        }
    
    with open(filename, 'w') as f:
        json.dump(path_data, f)
    
    return path_data

# 경로 데이터 불러오기
def load_path_data(filename='path_data.json'):
    with open(filename, 'r') as f:
        return json.load(f)

# 경로 시각화
def plot_paths(path_data):
    plt.figure(figsize=(12, 6))
    
    for name, data in path_data.items():
        plt.plot(data['t'], data['x'], label=name, linewidth=2)
    
    plt.title('Different Speed Profiles for Drone Movement')
    plt.xlabel('Time (t)')
    plt.ylabel('X Position')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig("two_paths.png")

# 메인 실행 부분
if __name__ == "__main__":
    # 경로 데이터 저장
    path_data = save_path_data()
    
    # 경로 데이터 불러오기
    loaded_path_data = load_path_data()
    
    # 경로 시각화
    plot_paths(loaded_path_data)
    
    # 특정 시간에서의 위치 값 얻기 예시
    print("\nExample position values at t=5.0:")
    t_index = int(len(t)/2)  # 5초 지점의 인덱스
    for name, data in loaded_path_data.items():
        print(f"{name}: x = {data['x'][t_index]:.2f}")