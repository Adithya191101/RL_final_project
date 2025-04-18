# Repository Structure

This document explains the organization of the RL-based robotic arm control project.

## Main Components

- `main.py`: Entry point for running training, evaluation, and benchmarking
- `configs/`: Configuration files for different training approaches
- `models/`: Directory for saved model checkpoints
- `logs/`: Directory for training logs, visualizations, and metrics
- `src/`: Source code for the project

## Source Code Structure

The `src/` directory contains all the implementation code:

### Controllers (`src/controllers/`)

- `hybrid_controller.py`: Combines RL-based and model-based control approaches
- `mpc_controller.py`: Model Predictive Control implementation

### Environment (`src/env/`)

- `robot_arm_env.py`: PyBullet-based robotic arm environment
- `camera.py`: Camera perception implementation

### Reinforcement Learning (`src/rl/`)

- `ppo_agent.py`: Proximal Policy Optimization implementation
- `smdp_agent.py`: Semi-Markov Decision Process implementation for temporal abstraction

### Utilities (`src/utils/`)

- `visualization.py`: Functions for plotting training progress and results

### Core Scripts

- `train.py`: Core training implementation
- `benchmark.py`: Benchmark different approaches (PPO, SMDP, MPC)

## Configuration Files

- `default.yaml`: Standard configuration settings
- `fixed_training.yaml`: Optimized parameters for more effective training

## Usage Patterns

1. **Training a model**:
   ```bash
   python main.py --mode train --config configs/fixed_training.yaml
   ```

2. **Evaluating a trained model**:
   ```bash
   python main.py --mode evaluate --model_path models/model_final.pt
   ```

3. **Running benchmarks**:
   ```bash
   python main.py --mode benchmark
   ```