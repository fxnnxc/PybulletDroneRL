#!/usr/bin/env python
import rospy
import json
from std_msgs.msg import String  # 예시로 String 메시지 타입 사용
from datetime import datetime

class MultiTopicToJson:
    def __init__(self):
        rospy.init_node('multi_topic_to_json_node', anonymous=True)
        
        # JSON 파일 경로 설정
        self.json_file = f"ros_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.data_dict = {}
        
        # 구독할 토픽 리스트
        self.topic_list = [
            '/rostopic/aaa',
            '/rostopic/bbb',
            # 필요한 만큼 토픽을 추가하세요
        ]
        
        # 각 토픽별 데이터 저장소 초기화
        for topic in self.topic_list:
            self.data_dict[topic] = []
            
        # 각 토픽에 대한 구독자 설정
        self.subscribers = [
            rospy.Subscriber(topic, String, self.callback, callback_args=topic)
            for topic in self.topic_list
        ]
        
    def callback(self, msg, topic_name):
        # 받은 메시지를 딕셔너리 형태로 변환
        data = {
            'timestamp': rospy.get_time(),
            'data': msg.data  # 실제 메시지 구조에 맞게 수정하세요
        }
        
        # 해당 토픽의 데이터 리스트에 추가채
        self.data_dict[topic_name].append(data)
        
        # JSON 파일로 저장
        self.save_to_json()
        
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
