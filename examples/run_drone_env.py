from gym_pybullet_drones.utils.enums import ObservationType, ActionType

DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('pid') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'
from env import WayPointAviary

env = WayPointAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT)
print(env)

obs, _ = env.reset()
print(obs)
print(env.action_space)
print(env.observation_space)

action = env.action_space.sample()
print(action)

state, reward, done, trun, info = env.step(action)
print(state)
print(reward)
print(done)
print(info)