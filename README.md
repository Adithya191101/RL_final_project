  # Robotic Arm Control with Reinforcement Learning

A PyBullet-based robotic arm control project using state-of-the-art reinforcement learning algorithms.

## Project Overview

This project implements a reinforcement learning framework for robotic arm trajectory planning and control. Using PyBullet for physics simulation, the project demonstrates how RL can be applied to teach a robotic arm to reach target positions while avoiding obstacles.

### Key Features

- **PPO-based Reinforcement Learning**: Proximal Policy Optimization for stable policy improvement
- **Semi-Markov Decision Process (SMDP)**: Temporal abstraction for complex action sequences
- **Model Predictive Control (MPC)**: Local trajectory refinement integration
- **Optimized Physics Parameters**: Calibrated for realistic and effective robot movement
- **PyBullet Integration**: High-performance physics simulation

## Project Structure

```
├── main.py                  # Main entry point for training and evaluation
├── configs/                 # Configuration files
│   ├── default.yaml         # Standard configuration
│   └── fixed_training.yaml  # Optimized parameters configuration
├── models/                  # Saved model checkpoints
├── logs/                    # Training logs and metrics
└── src/                     # Source code
    ├── controllers/         # Robot controllers
    │   ├── hybrid_controller.py   # Combined RL+MPC controller
    │   └── mpc_controller.py      # Model Predictive Control implementation
    ├── env/                 # Environment implementation
    │   ├── robot_arm_env.py     # PyBullet robotic arm environment
    │   └── camera.py            # Camera perception module
    ├── rl/                  # Reinforcement learning algorithms
    │   ├── ppo_agent.py     # PPO implementation
    │   └── smdp_agent.py    # Semi-MDP implementation
    ├── utils/               # Utility functions
    └── train.py             # Core training implementation
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Adithya191101/RL_final_project
   cd robotic-arm-rl
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

Train a model with the default or optimized parameters:

```bash
python main.py --mode train --config configs/fixed_training.yaml
```

Options:
- `--timesteps`: Number of timesteps (default: 200000)
- `--render`: Enable rendering to visualize training
- `--save_dir`: Directory to save models (default: "models")

### Evaluation

Evaluate a trained model:

```bash
python main.py --mode evaluate --model_path models/model_final.pt
```

Options:
- `--episodes`: Number of evaluation episodes (default: 10)
- `--render`: Enable rendering to visualize evaluation
- `--camera_view`: Enable camera view visualization

### Benchmarking

Compare different approaches (PPO, SMDP, MPC):

```bash
python main.py --mode benchmark
```

## Configuration

The project uses YAML configuration files with the following key sections:

- **environment**: Physics parameters, robot settings, and goals
- **rl**: Reinforcement learning algorithm settings
- **mpc**: Model predictive control parameters (when enabled)

Example configuration:
```yaml
environment:
  robot: "kuka"
  action_scale: 10.0
  action_force: 100.0
  goal_position: [0.2, 0.2, 0.3]
  goal_tolerance: 0.2

rl:
  algorithm: "ppo"
  learning_rate: 5.0e-4
  total_timesteps: 200000
```

## Implementation Details

### Reinforcement Learning Approach

The project uses PPO (Proximal Policy Optimization) as the primary RL algorithm, with extensions for:

1. **Temporal Abstraction**: Using SMDP for variable-duration actions
2. **Hybrid Control**: Combining RL with model-based approaches
3. **Effective Reward Design**: Progress rewards, goal bonuses, and movement incentives

### Robot Movement Optimization

Special attention has been paid to optimizing the robot's movement capabilities:

- Calibrated action scale and force parameters
- Stability improvements to prevent "falling down" issues
- Goal positioning within reachable workspace

## License

This project is licensed under the MIT License - see the LICENSE file for details.# RL_final_project
