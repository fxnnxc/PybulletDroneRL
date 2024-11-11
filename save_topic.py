#!/usr/bin/env python
import rospy
import json
from std_msgs.msg import String, Int32, Float32, Bool
from sensor_msgs.msg import Imu, NavSatFix, Image
from geometry_msgs.msg import Twist, Pose, PoseStamped
from nav_msgs.msg import Odometry
from datetime import datetime
import time
from collections import deque
from rospy_message_converter import message_converter
from sensor_msgs.msg import BatteryState

class MultiTopicToJson:
    def __init__(self):
        rospy.init_node('multi_topic_to_json_node', anonymous=True)
        
        # 토픽과 해당 메시지 타입을 딕셔너리로 정의
        self.topic_types = {
            '/rostopic/aaa': Imu,           # 예: IMU 데이터
            '/rostopic/bbb': Odometry,      # 예: 오도메트리 데이터
            '/rostopic/ccc': PoseStamped,   # 예: 위치 데이터
            '/battery': BatteryState,       # 배터리 상태 토픽 추가
            # 필요한 토픽과 타입을 추가하세요
        }
        
        self.sampling_rate = 5
        self.sampling_interval = 1.0 / self.sampling_rate
        self.recording_duration = 60
        self.auto_save_interval = 10
        self.last_auto_save = time.time()
        
        # 토픽 리스트 업데이트
        self.topic_list = list(self.topic_types.keys())
        
        # 타이머 설정
        rospy.Timer(rospy.Duration(1.0), self.print_status)
        rospy.Timer(rospy.Duration(self.sampling_interval), self.save_sample)
        rospy.Timer(rospy.Duration(self.auto_save_interval), self.auto_save)
        
        self.initialize_recording()

    def initialize_recording(self):
        self.json_file = f"ros_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.start_time = time.time()
        self.last_auto_save = time.time()
        
        self.temp_buffers = {}
        self.data_dict = {}
        self.message_counts = {}
        
        for topic in self.topic_list:
            self.data_dict[topic] = []
            self.temp_buffers[topic] = deque(maxlen=1)
            self.message_counts[topic] = 0
            
        if hasattr(self, 'subscribers'):
            for sub in self.subscribers:
                sub.unregister()
                
        # 각 토픽 타입에 맞는 Subscriber 생성
        self.subscribers = [
            rospy.Subscriber(topic, msg_type, self.callback, callback_args=topic)
            for topic, msg_type in self.topic_types.items()
        ]
        
        print("\n=== New Recording Session Started ===")
        print(f"Sampling Rate: {self.sampling_rate} Hz")
        print(f"Recording Duration: {self.recording_duration} seconds")
        print(f"Auto-save interval: {self.auto_save_interval} seconds")
        print(f"Output file: {self.json_file}")
        print("\nSubscribing to topics:")
        for topic, msg_type in self.topic_types.items():
            print(f"- {topic} ({msg_type.__name__})")

    def convert_msg_to_dict(self, msg):
        """ROS 메시지를 딕셔너리로 변환"""
        if isinstance(msg, (int, float, str, bool)):
            return msg
            
        if hasattr(msg, 'header'):
            # 헤더가 있는 메시지 처리
            data = {
                'header': {
                    'seq': msg.header.seq,
                    'stamp': msg.header.stamp.to_sec(),
                    'frame_id': msg.header.frame_id
                }
            }
        else:
            data = {}

        # 메시지 타입별 처리
        if isinstance(msg, Imu):
            data.update({
                'orientation': {'x': msg.orientation.x, 'y': msg.orientation.y, 
                              'z': msg.orientation.z, 'w': msg.orientation.w},
                'angular_velocity': {'x': msg.angular_velocity.x, 'y': msg.angular_velocity.y, 
                                   'z': msg.angular_velocity.z},
                'linear_acceleration': {'x': msg.linear_acceleration.x, 'y': msg.linear_acceleration.y, 
                                      'z': msg.linear_acceleration.z}
            })
        elif isinstance(msg, Odometry):
            data.update({
                'pose': {
                    'position': {'x': msg.pose.pose.position.x, 'y': msg.pose.pose.position.y, 
                               'z': msg.pose.pose.position.z},
                    'orientation': {'x': msg.pose.pose.orientation.x, 'y': msg.pose.pose.orientation.y,
                                  'z': msg.pose.pose.orientation.z, 'w': msg.pose.pose.orientation.w}
                },
                'twist': {
                    'linear': {'x': msg.twist.twist.linear.x, 'y': msg.twist.twist.linear.y, 
                             'z': msg.twist.twist.linear.z},
                    'angular': {'x': msg.twist.twist.angular.x, 'y': msg.twist.twist.angular.y, 
                              'z': msg.twist.twist.angular.z}
                }
            })
        elif isinstance(msg, PoseStamped):
            data.update({
                'pose': {
                    'position': {'x': msg.pose.position.x, 'y': msg.pose.position.y, 
                               'z': msg.pose.position.z},
                    'orientation': {'x': msg.pose.orientation.x, 'y': msg.pose.orientation.y,
                                  'z': msg.pose.orientation.z, 'w': msg.pose.orientation.w}
                }
            })
        elif isinstance(msg, BatteryState):
            data.update({
                'voltage': msg.voltage,
                'current': msg.current,
                'percentage': msg.percentage,
                'power_supply_status': msg.power_supply_status,
                'power_supply_health': msg.power_supply_health,
                'present': msg.present,
                'cell_voltage': msg.cell_voltage,
                'location': msg.location
            })
        
        return data

    def callback(self, msg, topic_name):
        self.message_counts[topic_name] += 1
        
        # ROS 메시지를 딕셔너리로 변환하여 저장
        msg_dict = self.convert_msg_to_dict(msg)
        msg_dict['timestamp'] = rospy.get_time()
        
        self.temp_buffers[topic_name].append(msg_dict)

    def auto_save(self, event):
        """주기적으로 데이터 저장"""
        current_time = time.time()
        if current_time - self.last_auto_save >= self.auto_save_interval:
            self.save_to_json()
            self.last_auto_save = current_time
            print(f"\n[Auto-save] Data saved to {self.json_file}")

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
