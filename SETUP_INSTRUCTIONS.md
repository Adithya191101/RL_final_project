# Setup Instructions

This document provides instructions for setting up the project repository for GitHub.

## Repository Creation

1. Create a new repository on GitHub:
   - Go to https://github.com/new
   - Name the repository "RoboticArm-RL" or something similar
   - Add a description (e.g., "Reinforcement Learning for Robotic Arm Control")
   - Make it public or private as desired
   - Initialize without README.md (we already have one)

2. Initialize the local repository and push to GitHub:
   ```bash
   cd /home/adithya/RL_final_project
   git init
   git add .
   git commit -m "Initial commit: robotic arm control with reinforcement learning"
   git branch -M main
   git remote add origin https://github.com/yourusername/RoboticArm-RL.git
   git push -u origin main
   ```

## Running the Project

### Prerequisites

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Verify dependencies:
   ```bash
   python check_dependencies.py
   ```

### Training a Model

```bash
python main.py --mode train --config configs/fixed_training.yaml
```

Options:
- `--render`: Enable rendering
- `--timesteps`: Set number of training timesteps

### Evaluating a Model

```bash
python main.py --mode evaluate --model_path models/model_final.pt
```

Options:
- `--render`: Enable rendering
- `--episodes`: Set number of evaluation episodes

### Running Benchmarks

```bash
python main.py --mode benchmark
```

## Project Highlights

This repository implements:
1. RL-based robotic arm control using PPO
2. Temporal abstraction with SMDP
3. Hybrid control with MPC
4. PyBullet physics simulation

The optimized parameters ensure the robot can effectively move and reach goals, preventing the "falling down" problem by using higher action scales and forces.