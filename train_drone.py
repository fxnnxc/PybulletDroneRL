import argparse
import torch
import numpy as np 
import gymnasium as gym
import torch.nn as nn
import torch.optim as optim
from sac import Actor, SoftQNetwork, train_sac
from env import BatteryWayPointAviary
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
import os
from datetime import datetime
import random

def set_seed(seed):
    """Set all random seeds for reproducibility"""
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--battery-alpha", type=float, default=0.1,
                        help="Battery consumption penalty coefficient (default: 0.1)")
    parser.add_argument("--total-timesteps", type=int, default=100000,
                        help="Total timesteps for training (default: 100000)")
    parser.add_argument("--learning-starts", type=int, default=1000,
                        help="Number of timesteps before learning starts (default: 1000)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run training on (cuda or cpu)")
    parser.add_argument("--exp-name", type=str, default="drone_sac",
                        help="Name of the experiment")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    parser.add_argument("--hidden-size", type=int, default=256,
                        help="Hidden layer size for networks (default: 256)")
    parser.add_argument("--save-dir", type=str, default="outputs",
                        help="Directory to save models and tensorboard logs")
    parser.add_argument("--seed", type=int, default=1,
                        help="Random seed (default: 1)")
    args = parser.parse_args()
    return args

class CustomEnv():
    def __init__(self, battery_alpha):

        DEFAULT_OBS = ObservationType('kin') # 'kin' or 'rgb'
        DEFAULT_ACT = ActionType('pid') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'

        self.env = BatteryWayPointAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT, battery_alpha=battery_alpha)
        
        action_dim = 3
        obs_dim = self.env.observation_space.shape[-1]
        
        print("=======Pybullet============")
        print(self.env.action_space)
        print(self.env.observation_space.shape)
        print("===========================")
        
        
        self.single_action_space  = gym.spaces.Box(-1, 1, shape=(action_dim,))
        self.single_observation_space  = gym.spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)
        
        
    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
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
    


def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create environment
    env = CustomEnv(args.battery_alpha)
    
    # Get dimensions
    obs_dim = [env.single_observation_space.shape[0],]
    action_dim = [env.single_action_space.shape[0],]
    
    # Get action space bounds
    action_space_low = env.single_action_space.low
    action_space_high = env.single_action_space.high
    
    # Create timestamp string
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create run name with timestamp and battery_alpha
    run_name = f"battery_{args.battery_alpha}_seed{args.seed}/{current_time}"
    
    # Create full save path
    save_path = os.path.join(args.save_dir, run_name)
    os.makedirs(save_path, exist_ok=True)
    
    print("\n====== Training Settings ======")
    print(f"Save path: {save_path}")
    print(f"Seed: {args.seed}")
    print(f"Battery Alpha: {args.battery_alpha}")
    print(f"Total Timesteps: {args.total_timesteps}")
    print(f"Learning Starts: {args.learning_starts}")
    print(f"Device: {args.device}")
    print("=============================\n")
    
    # Initialize networks
    actor = Actor(obs_dim, action_dim, action_space_high, action_space_low).to(args.device)
    qf1 = SoftQNetwork(obs_dim, action_dim).to(args.device)
    qf2 = SoftQNetwork(obs_dim, action_dim).to(args.device)
    qf1_target = SoftQNetwork(obs_dim, action_dim).to(args.device)
    qf2_target = SoftQNetwork(obs_dim, action_dim).to(args.device)
    
    # Copy parameters
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    
    # Train with save directory
    train_sac(
        envs=env,
        actor=actor,
        qf1=qf1,
        qf2=qf2,
        qf1_target=qf1_target,
        qf2_target=qf2_target,
        device=args.device,
        total_timesteps=args.total_timesteps,
        learning_starts=args.learning_starts,
        save_dir=save_path
    )

if __name__ == "__main__":
    main()