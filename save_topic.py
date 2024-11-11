#!/usr/bin/env python
import rospy
import json
from std_msgs.msg import String
from datetime import datetime
import time
from collections import deque

class MultiTopicToJson:
    def __init__(self):
        rospy.init_node('multi_topic_to_json_node', anonymous=True)
        
        # 샘플링 설정
        self.sampling_rate = 5  # 1초당 저장할 샘플 수
        self.sampling_interval = 1.0 / self.sampling_rate
        self.recording_duration = 60  # 각 파일당 녹화 시간 (초)
        self.auto_save_interval = 10  # 자동 저장 주기 (초)
        self.last_auto_save = time.time()
        
        # 구독할 토픽 리스트
        self.topic_list = [
            '/rostopic/aaa',
            '/rostopic/bbb',
        ]
        
        # 타이머 설정
        rospy.Timer(rospy.Duration(1.0), self.print_status)
        rospy.Timer(rospy.Duration(self.sampling_interval), self.save_sample)
        rospy.Timer(rospy.Duration(self.auto_save_interval), self.auto_save)  # 주기적 저장 타이머
        
        self.initialize_recording()
        
    def initialize_recording(self):
        """새로운 녹화 세션 초기화"""
        self.json_file = f"ros_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.start_time = time.time()
        self.last_auto_save = time.time()
        
        # 버퍼 및 데이터 저장소 초기화
        self.temp_buffers = {}
        self.data_dict = {}
        self.message_counts = {}
        
        for topic in self.topic_list:
            self.data_dict[topic] = []
            self.temp_buffers[topic] = deque(maxlen=1)
            self.message_counts[topic] = 0
            
        # 구독자 재설정
        if hasattr(self, 'subscribers'):
            for sub in self.subscribers:
                sub.unregister()
                
        self.subscribers = [
            rospy.Subscriber(topic, String, self.callback, callback_args=topic)
            for topic in self.topic_list
        ]
        
        print("\n=== New Recording Session Started ===")
        print(f"Sampling Rate: {self.sampling_rate} Hz")
        print(f"Recording Duration: {self.recording_duration} seconds")
        print(f"Auto-save interval: {self.auto_save_interval} seconds")
        print(f"Recording will end at: {datetime.fromtimestamp(time.time() + self.recording_duration).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Output file: {self.json_file}")
        print("\nSubscribing to topics:")
        for topic in self.topic_list:
            print(f"- {topic}")
        print("\nRecording... Press Ctrl+C to stop completely\n")

    def auto_save(self, event):
        """주기적으로 데이터 저장"""
        current_time = time.time()
        if current_time - self.last_auto_save >= self.auto_save_interval:
            self.save_to_json()
            self.last_auto_save = current_time
            print(f"\n[Auto-save] Data saved to {self.json_file}")

    def callback(self, msg, topic_name):
        self.message_counts[topic_name] += 1
        self.temp_buffers[topic_name].append({
            'timestamp': rospy.get_time(),
            'data': msg.data
        })
        
    def print_status(self, event):
        elapsed_time = time.time() - self.start_time
        remaining_time = max(0, self.recording_duration - elapsed_time)
        next_save = self.auto_save_interval - (time.time() - self.last_auto_save)
        
        print("\n--- Recording Status ---")
        print(f"Current file: {self.json_file}")
        print(f"Elapsed Time: {int(elapsed_time)}s / {self.recording_duration}s")
        print(f"Remaining Time: {int(remaining_time)}s")
        print(f"Next auto-save in: {int(next_save)}s")
        print("Message counts:")
        for topic, count in self.message_counts.items():
            print(f"- {topic}: {count} messages")
        print("----------------------")
        
    def save_sample(self, event):
        current_time = time.time()
        
        # 현재 녹화 세션의 시간이 다 되었는지 체크
        if current_time - self.start_time > self.recording_duration:
            self.save_to_json()
            print(f"\n=== Recording Session Completed ===")
            print(f"Data saved to: {self.json_file}")
            print("Final message counts:")
            for topic, count in self.message_counts.items():
                print(f"- {topic}: {count} messages")
            print("=========================")
            
            # 새로운 녹화 세션 시작
            self.initialize_recording()
            return
            
        # 각 토픽에 대해 샘플링 간격마다 데이터 저장
        for topic in self.topic_list:
            if self.temp_buffers[topic]:
                latest_data = self.temp_buffers[topic][-1]
                self.data_dict[topic].append(latest_data)
        
    def save_to_json(self):
        try:
            with open(self.json_file, 'w') as f:
                json.dump(self.data_dict, f, indent=4)
        except Exception as e:
            print(f"\n[Error] Failed to save data: {str(e)}")
            
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = MultiTopicToJson()
        node.run()
    except rospy.ROSInterruptException:
        pass
