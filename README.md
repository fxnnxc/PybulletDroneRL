# PybulletDroneRL

A reinforcement learning environment for drone navigation using PyBullet.

## Quick Start

```bash
python train_drone.py
```

## Environment Details

### Observation Space
The observation space consists of:
- Current position (x, y, z)
- Current quaternion orientation (x, y, z, w)
- Current linear velocity (vx, vy, vz)
- Current angular velocity (wx, wy, wz)
- Relative distance to target waypoint (dx, dy, dz)

### Action Space
The action space represents position adjustments:
- Actions are sampled in the range [-1, 1] for each dimension (x, y, z)
- The sampled action is added to the current target position to generate a refined waypoint
- This refined waypoint serves as the reference for the drone's low-level controller

### Reward Structure
The reward function consists of three components:

1. Waypoint Achievement Reward (Positive)
   - +10 when the drone successfully reaches a waypoint
   - Encourages completion of navigation objectives

2. Distance-based Reward (Positive)
   - Calculated as 1/(1 + distance²)
   - Provides continuous feedback for approaching the target
   - Normalized between 0 and 1

3. Battery Constraint Penalty (Negative)
   - Penalizes excessive energy consumption
   - Encourages efficient path planning

### Environment Initialization
The environment initializes with:
- Random starting position within defined bounds
- Zero initial orientation (quaternion)
- Zero initial linear and angular velocities
- Calculates initial relative distance to the target waypoint

## Dependencies
- PyBullet
- OpenAI Gym
- NumPy
- PyTorch (for training)


