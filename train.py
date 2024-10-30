from sac import Actor, SoftQNetwork, train_sac
from env import WayPointAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

import numpy as np 
import gymnasium as gym
class CustomEnv():
    def __init__(self):

        DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
        DEFAULT_ACT = ActionType('pid') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'

        self.env = WayPointAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
        
        action_dim = 3
        obs_dim = self.env.observation_space.shape[-1]
        
        print("=======Pybullet============")
        print(self.env.action_space)
        print(self.env.observation_space.shape)
        print("===========================")
        
        
        self.single_action_space  = gym.spaces.Box(-np.inf, np.inf, shape=(action_dim,))
        self.single_observation_space  = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        
        
    def step(self, action):
        
        if self.done:
            obs, info = self.env.reset()
            self.done = False 
            return obs, 0, False, False, info 

        if action.ndim == 3:
            action = action[0]
        state, reward, done, truc, info = self.env.step(action)
        self.info['episode']['l'] += 1 
        self.info['episode']['r'] += reward
        self.done = done

        return [state], [reward], [done], [truc], [self.info] 
    
    def reset(self, seed=0):
        self.done = False 
        state, _= self.env.reset()
        self.info = {
            'episode':{
                'r': 0,
                'l': 0,
            }
        }
        return state, self.info 
    
envs = CustomEnv()

observation_shape = [envs.single_observation_space.shape[-1],]
action_shape = [envs.single_action_space.shape[-1],]
action_space_high =[1,1,1]
action_space_low  =[-1,-1,-1]
actor = Actor(observation_shape, action_shape, action_space_high, action_space_low)

import copy
qf1 = SoftQNetwork(observation_shape, action_shape)
qf2 = SoftQNetwork(observation_shape, action_shape)
qf1_target = copy.deepcopy(qf1)
qf2_target = copy.deepcopy(qf2)



train_sac(envs, actor, qf1, qf2, qf1_target, qf2_target, run_name='test', 
          device='cuda',
          total_timesteps=100000,
          learning_starts=1000,
)