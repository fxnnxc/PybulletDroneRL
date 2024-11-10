import torch
import numpy as np
from env import BatteryWayPointAviary
from sac import Actor
from gym_pybullet_drones.utils.enums import ObservationType, ActionType
import gymnasium as gym
from tqdm import tqdm
import os
import json
from datetime import datetime
from train_drone import CustomEnv   

def test(model_path, n_episodes=100, device="cuda"):
    # Create results directory
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path).split('.')[0]  # e.g., 'actor_90000'
    results_dir = os.path.join(model_dir, 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize environment and model
    env = CustomEnv(battery_alpha=0.0)
    
    # Get dimensions and action space bounds
    obs_dim = [env.single_observation_space.shape[0],]
    action_dim = [env.single_action_space.shape[0],]
    action_space_high = env.single_action_space.high
    action_space_low = env.single_action_space.low
    
    # Initialize actor and load weights
    actor = Actor(obs_dim, action_dim, action_space_high, action_space_low).to(device)
    actor.load_state_dict(torch.load(model_path))
    actor.eval()
    
    # Test loop
    episode_rewards = []
    episode_lengths = []
    waypoint_progress = []  # Track waypoint progress for each episode
    success_count = 0
    
    for episode in tqdm(range(n_episodes), desc="Testing episodes"):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        final_waypoint_idx = 0
        
        while not done:
            # Get action from policy
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).to(device)
                action = actor.get_action(state_tensor)[0]
                action = action.cpu().numpy()
            
            # Take step in environment
            next_state, reward, done, _, info = env.step(action)
            episode_reward += reward[0]
            episode_length += 1
            state = next_state[0]
            done = done[0]
            final_waypoint_idx = max(env.env.current_waypoint_idx, final_waypoint_idx)

        
        # Calculate waypoint progress ratio
        total_waypoints = len(env.env.waypoints)
        waypoint_ratio = final_waypoint_idx / total_waypoints
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        waypoint_progress.append(waypoint_ratio)
    
    # Calculate statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    success_rate = (success_count / n_episodes) * 100
    mean_length = np.mean(episode_lengths)
    std_length = np.std(episode_lengths)
    mean_waypoint_progress = np.mean(waypoint_progress)
    std_waypoint_progress = np.std(waypoint_progress)
    
    # Create results dictionary
    results = {
        "model_checkpoint": model_name,
        "evaluation_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "episodes": n_episodes,
        "statistics": {
            "mean_reward": float(mean_reward),
            "std_reward": float(std_reward),
            "success_rate": float(success_rate),
            "mean_episode_length": float(mean_length),
            "std_episode_length": float(std_length),
            "mean_waypoint_progress": float(mean_waypoint_progress),
            "std_waypoint_progress": float(std_waypoint_progress)
        },
        "episode_details": {
            "rewards": [float(r) for r in episode_rewards],
            "lengths": episode_lengths,
            "waypoint_progress": [float(w) for w in waypoint_progress]
        }
    }
    
    # Save results
    results_file = os.path.join(results_dir, f"{model_name}_eval_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Print summary
    print("\n====== Evaluation Results ======")
    print(f"Model: {model_name}")
    print(f"Episodes: {n_episodes}")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Mean Episode Length: {mean_length:.2f} ± {std_length:.2f}")
    print(f"Mean Waypoint Progress: {mean_waypoint_progress:.2f} ± {std_waypoint_progress:.2f}")
    print(f"Results saved to: {results_file}")
    print("==============================\n")
    
    return mean_reward, std_reward, success_rate, mean_waypoint_progress

if __name__ == "__main__":
    model_path = "outputs/battery_0.0_seed1/20241111_012334/actor_180000.pth"
    test(model_path, n_episodes=10)