"""
Training script for the RL-based robotic arm trajectory planning system.
"""
import os
import time
import yaml
import argparse
import numpy as np
from tqdm import tqdm
import sys
import gym

# Set matplotlib backend to Agg to avoid display issues
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Check CUDA availability
import torch
if torch.cuda.is_available():
    print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
    # Set CUDA to be visible
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
    print("CUDA is not available, using CPU")

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Use relative imports
from env.robot_arm_env import RobotArmEnv
from controllers.hybrid_controller import HybridController


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train the RL-based arm controller")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--model_dir", type=str, default="models", help="Directory to save models"
    )
    parser.add_argument(
        "--log_dir", type=str, default="logs", help="Directory to save logs"
    )
    parser.add_argument(
        "--render", action="store_true", help="Render training"
    )
    parser.add_argument(
        "--eval", action="store_true", help="Evaluate instead of train"
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to model for evaluation"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=1000, help="Number of episodes to train/evaluate"
    )
    return parser.parse_args()


def train(config, args):
    """
    Train the hybrid controller.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Set rendering based on args
    config["environment"]["render"] = args.render
    
    # Create environment
    env = RobotArmEnv(config)
    
    # Create controller
    state_dim = env.observation_space.shape[0]
    
    # Handle different action space types
    if isinstance(env.action_space, gym.spaces.Dict):
        # For SMDP, use the joint_velocities shape
        action_dim = env.action_space['joint_velocities'].shape[0]
    else:
        # For standard MDP
        action_dim = env.action_space.shape[0]
        
    controller = HybridController(config, state_dim, action_dim)
    
    # Load model if specified
    if args.model_path is not None:
        controller.load_model(args.model_path)
        print(f"Loaded model from {args.model_path}")
    
    # Training loop
    total_timesteps = config["rl"]["total_timesteps"]
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    
    # Progress bar
    pbar = tqdm(total=total_timesteps)
    
    # Variables for tracking
    timesteps_so_far = 0
    best_success_rate = 0.0
    
    while timesteps_so_far < total_timesteps:
        # Reset environment
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        # Episode loop
        while not done:
            # Select action (using exploration during training)
            action = controller.select_action(state, env, deterministic=False)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Update controller with real experience
            controller.update_with_real_experience(state, action, reward, next_state, done, info)
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            episode_length += 1
            timesteps_so_far += 1
            pbar.update(1)
            
            # Break if max timesteps reached
            if timesteps_so_far >= total_timesteps:
                break
        
        # Record episode stats
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_successes.append(1 if info.get("goal_reached", False) else 0)
        
        # Logging
        if len(episode_rewards) % 10 == 0:
            recent_rewards = episode_rewards[-10:]
            recent_lengths = episode_lengths[-10:]
            recent_successes = episode_successes[-10:]
            
            avg_reward = np.mean(recent_rewards)
            avg_length = np.mean(recent_lengths)
            success_rate = np.mean(recent_successes)
            
            print(f"\nEpisode {len(episode_rewards)}, Timestep {timesteps_so_far}/{total_timesteps}")
            print(f"Average Reward: {avg_reward:.2f}")
            print(f"Average Length: {avg_length:.2f}")
            print(f"Success Rate: {success_rate:.2f}")
            print(f"Controller: {controller.current_controller}")
            
            # Save model if success rate improved
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                model_path = os.path.join(args.model_dir, f"model_best.pt")
                controller.save_model(model_path)
                print(f"Saved best model with success rate {best_success_rate:.2f}")
        
        # Save model at percentage-based checkpoints
        progress_percentage = min(100, int((timesteps_so_far / total_timesteps) * 100))
        
        # Define percentage checkpoints (0%, 25%, 50%, 75%, 100%)
        checkpoints = [0, 25, 50, 75, 100]
        
        # Calculate closest lower checkpoint
        current_checkpoint = max([p for p in checkpoints if p <= progress_percentage])
        
        # Check if we've reached a new checkpoint
        if hasattr(train, 'last_checkpoint'):
            if current_checkpoint > train.last_checkpoint:
                # Save model at the new checkpoint
                model_path = os.path.join(args.model_dir, f"model_{current_checkpoint}pct.pt")
                controller.save_model(model_path)
                print(f"Saved {current_checkpoint}% checkpoint model to {model_path}")
                
                # Update the last checkpoint
                train.last_checkpoint = current_checkpoint
                
                # Plot and save learning curves
                plot_learning_curves(episode_rewards, episode_lengths, episode_successes, args.log_dir)
        else:
            # Initialize the last checkpoint attribute
            train.last_checkpoint = current_checkpoint
            
            # Save the 0% checkpoint (initial model)
            if current_checkpoint == 0:
                model_path = os.path.join(args.model_dir, "model_0pct.pt")
                controller.save_model(model_path)
                print(f"Saved 0% checkpoint model to {model_path}")
    
    # Close environment
    env.close()
    pbar.close()
    
    # Final save
    model_path = os.path.join(args.model_dir, "model_100pct.pt")
    controller.save_model(model_path)
    print(f"Saved final model (100%) to {model_path}")
    
    # Also save as model_final.pt for backward compatibility
    final_path = os.path.join(args.model_dir, "model_final.pt")
    controller.save_model(final_path)
    
    # Final plots
    plot_learning_curves(episode_rewards, episode_lengths, episode_successes, args.log_dir)


def evaluate(config, args):
    """
    Evaluate the hybrid controller.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    # Set rendering based on args
    config["environment"]["render"] = args.render
    
    # Create environment
    env = RobotArmEnv(config)
    
    # Create controller
    state_dim = env.observation_space.shape[0]
    
    # Handle different action space types
    if isinstance(env.action_space, gym.spaces.Dict):
        # For SMDP, use the joint_velocities shape
        action_dim = env.action_space['joint_velocities'].shape[0]
    else:
        # For standard MDP
        action_dim = env.action_space.shape[0]
        
    controller = HybridController(config, state_dim, action_dim)
    
    # Load model
    if args.model_path is None:
        raise ValueError("Model path is required for evaluation")
    
    controller.load_model(args.model_path)
    print(f"Loaded model from {args.model_path}")
    
    # Evaluation loop
    num_episodes = args.num_episodes
    episode_rewards = []
    episode_lengths = []
    episode_successes = []
    controller_usage = {"rl": 0, "mpc": 0}
    
    for episode in range(num_episodes):
        # Reset environment
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        # Episode loop
        while not done:
            # Select action (use deterministic policy for evaluation)
            action = controller.select_action(state, env, deterministic=True)
            
            # Track controller usage
            controller_usage[controller.current_controller] += 1
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # Sleep for visualization
            if args.render:
                time.sleep(0.01)
        
        # Record episode stats
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        episode_successes.append(1 if info.get("goal_reached", False) else 0)
        
        # Logging
        print(f"Episode {episode+1}/{num_episodes}")
        print(f"Reward: {episode_reward:.2f}")
        print(f"Length: {episode_length}")
        print(f"Goal Reached: {info.get('goal_reached', False)}")
        print(f"Collision: {info.get('collision', False)}")
    
    # Print evaluation results
    success_rate = np.mean(episode_successes)
    avg_reward = np.mean(episode_rewards)
    avg_length = np.mean(episode_lengths)
    
    print("\nEvaluation Results:")
    print(f"Success Rate: {success_rate:.2f}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Length: {avg_length:.2f}")
    print(f"Controller Usage - RL: {controller_usage['rl']}, MPC: {controller_usage['mpc']}")
    
    # Close environment
    env.close()


def plot_learning_curves(rewards, lengths, successes, log_dir):
    """
    Plot and save learning curves.
    
    Args:
        rewards: List of episode rewards
        lengths: List of episode lengths
        successes: List of episode successes (0 or 1)
        log_dir: Directory to save plots
    """
    try:
        # Use non-interactive Agg backend to avoid display issues
        import matplotlib
        matplotlib.use('Agg')
        
        # Create figure with 3 subplots
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        
        # Plot rewards
        axes[0].plot(rewards, 'b-')
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Reward')
        axes[0].set_title('Episode Rewards')
        
        # Plot episode lengths
        axes[1].plot(lengths, 'r-')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Length')
        axes[1].set_title('Episode Lengths')
        
        # Plot success rate (moving average)
        window_size = min(100, len(successes))
        if window_size > 0:
            moving_avg = np.convolve(successes, np.ones(window_size)/window_size, mode='valid')
            axes[2].plot(moving_avg, 'g-')
            axes[2].set_xlabel('Episode')
            axes[2].set_ylabel('Success Rate')
            axes[2].set_title(f'Success Rate (Moving Average, Window={window_size})')
            axes[2].set_ylim([0, 1])
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'learning_curves.png'))
        plt.close()
        
        print(f"Learning curves saved to {os.path.join(log_dir, 'learning_curves.png')}")
    except Exception as e:
        print(f"Error plotting learning curves: {e}")
        print("Training completed successfully but could not generate plots.")


def main():
    """Main function."""
    # Parse args
    args = parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Train or evaluate
    if args.eval:
        evaluate(config, args)
    else:
        train(config, args)


if __name__ == "__main__":
    main()