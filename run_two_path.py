import json
import time
import numpy as np

def get_path_position(path_data, path_name, index):
    """주어진 시간에 해당하는 x 위치를 반환"""
    t_array = np.array(path_data[path_name]['t'])
    x_array = np.array(path_data[path_name]['x'])
    
    # 현재 시간이 범위를 벗어나면 처리
    if index >= len(t_array):
        return x_array[-1]
    else:
        return x_array[index]

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run two paths')
    parser.add_argument('--name', type=int, default=0)
    parser.add_argument('--tt', type=float, default=20)
    parser.add_argument('--dis', type=float, default=2.0)
    args = parser.parse_args()
    # 경로 이름 설정
    NAMES = ['constant_speed', 
             'fast_then_slow1',
            'fast_then_slow2',
            'fast_then_slow3',
            'slow_then_fast1',
            'slow_then_fast2',
            'slow_then_fast3',
            'periodic_speed1',
            'periodic_speed2']
    # NAME = 'constant_speed'
    NAME = NAMES[args.name]
    target_time=args.tt
    distance_increase_ratio=args.dis  # 1.0 = 10 m
    
    # 경로 데이터 불러오기
    try:
        with open('path_data.json', 'r') as f:
            path_data = json.load(f)
    except FileNotFoundError:
        print("Error: path_data.json file not found!")
        return
    
    # 시작 시간 기록
    start_time = time.time()
    last_time = start_time
    index =0 
    x_pos= path_data[NAME]['x'][0]
    
    interval=target_time/len(path_data[NAME]['t'])
    while index < len(path_data[NAME]['t']):
        # 현재 경과 시간 계산
        current_time = time.time()
        
        # x 위치 계산
        if current_time - last_time > interval:
            index = index + 1
            x_pos = get_path_position(path_data, NAME, index) * distance_increase_ratio
            last_time = current_time
            print(f"Time: {NAME} {current_time - start_time:.2f}/{target_time} s, X Position: {x_pos:.3f}")
        time.sleep(0.01)        
    print(f"{NAME} completed!")
    
        

if __name__ == "__main__":
    main()