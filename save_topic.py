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
        self.sampling_interval = 1.0 / self.sampling_rate  # 샘플링 간격 (초)
        self.recording_duration = 6000  # 총 녹화 시간 (초)
        
        # 임시 버퍼 설정
        self.temp_buffers = {}
        
        self.json_file = f"ros_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.data_dict = {}
        self.last_save_time = {}
        
        # 구독할 토픽 리스트
        self.topic_list = [
            '/rostopic/aaa',
            '/rostopic/bbb',
        ]
        
        # 초기화
        for topic in self.topic_list:
            self.data_dict[topic] = []
            self.temp_buffers[topic] = deque(maxlen=1)  # 최신 데이터만 유지
            self.last_save_time[topic] = time.time()
            
        self.start_time = time.time()
        
        # 구독자 설정
        self.subscribers = [
            rospy.Subscriber(topic, String, self.callback, callback_args=topic)
            for topic in self.topic_list
        ]
        
        # 타이머 설정 (샘플링 간격으로 저장)
        rospy.Timer(rospy.Duration(self.sampling_interval), self.save_sample)
        
    def callback(self, msg, topic_name):
        # 최신 데이터를 임시 버퍼에 저장
        self.temp_buffers[topic_name].append({
            'timestamp': rospy.get_time(),
            'data': msg.data
        })
        
    def save_sample(self, event):
        current_time = time.time()
        
        # 녹화 시간 체크
        if current_time - self.start_time > self.recording_duration:
            self.save_to_json()
            rospy.signal_shutdown("Recording completed")
            return
            
        # 각 토픽에 대해 샘플링 간격마다 데이터 저장
        for topic in self.topic_list:
            if self.temp_buffers[topic]:  # 버퍼에 데이터가 있는 경우
                # 가장 최신 데이터 가져오기
                latest_data = self.temp_buffers[topic][-1]
                self.data_dict[topic].append(latest_data)
                
    def save_to_json(self):
        with open(self.json_file, 'w') as f:
            json.dump(self.data_dict, f, indent=4)
            
    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        node = MultiTopicToJson()
        node.run()
    except rospy.ROSInterruptException:
        pass
