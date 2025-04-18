"""
Benchmark module for comparing different approaches:
- Standard PPO
- PPO with SMDP
- PPO with SMDP and MPC
"""
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

from .env.robot_arm_env import RobotArmEnv
from .controllers.hybrid_controller import HybridController
from .train import train, evaluate

def run_benchmark(args):
    """
    Run a benchmark comparison of different approaches.
    
    Args:
        args: Command line arguments
    """
    # Create directories
    os.makedirs("logs/benchmark", exist_ok=True)
    os.makedirs("models/benchmark", exist_ok=True)
    
    # Load base configuration
    with open(args.config, "r") as f:
        base_config = yaml.safe_load(f)
        
    # Common settings across configs
    base_config["environment"]["render"] = args.render
    base_config["rl"]["total_timesteps"] = args.timesteps or 50000
    
    # Define the approaches to benchmark
    approaches = [
        {
            "name": "ppo",
            "config": {
                "environment": {
                    "use_smdp": False,
                    "use_mpc": False
                }
            },
            "save_path": "models/benchmark/benchmark_ppo.pt",
            "log_path": "logs/benchmark/benchmark_ppo_training.jsonl"
        },
        {
            "name": "ppo_smdp",
            "config": {
                "environment": {
                    "use_smdp": True,
                    "use_mpc": False
                }
            },
            "save_path": "models/benchmark/benchmark_ppo_smdp.pt",
            "log_path": "logs/benchmark/benchmark_ppo_smdp_training.jsonl"
        },
        {
            "name": "ppo_smdp_mpc",
            "config": {
                "environment": {
                    "use_smdp": True,
                    "use_mpc": True
                }
            },
            "save_path": "models/benchmark/benchmark_ppo_smdp_mpc.pt",
            "log_path": "logs/benchmark/benchmark_ppo_smdp_mpc_training.jsonl"
        }
    ]
    
    # Run training and evaluation for each approach
    results = {}
    
    for approach in approaches:
        print(f"\n{'='*80}")
        print(f"Benchmarking approach: {approach['name']}")
        print(f"{'='*80}")
        
        # Create config for this approach
        config = base_config.copy()
        
        # Update with approach-specific settings
        for key, value in approach["config"].items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value
                
        # Create modified args for this approach
        modified_args = type('Args', (), {})()
        for attr_name, attr_value in vars(args).items():
            setattr(modified_args, attr_name, attr_value)
            
        modified_args.model_dir = "models/benchmark"
        modified_args.log_dir = "logs/benchmark"
        modified_args.model_path = approach["save_path"]
        
        # Train the agent
        train(config, modified_args)
        
        # Evaluate the agent
        modified_args.num_episodes = 10  # Set to a fixed number for comparison
        metrics = evaluate(config, modified_args, return_metrics=True)
        
        # Store results
        results[approach["name"]] = metrics
    
    # Generate comparison plots
    generate_comparison_plots(results, "logs/benchmark")
    
    print("\nBenchmark complete. Results saved to logs/benchmark")
    
def generate_comparison_plots(results, output_dir):
    """
    Generate comparison plots between different approaches.
    
    Args:
        results: Dictionary of results by approach
        output_dir: Directory to save plots
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract metrics for plotting
    approaches = list(results.keys())
    success_rates = [results[a]["success_rate"] for a in approaches]
    avg_rewards = [results[a]["avg_reward"] for a in approaches]
    avg_lengths = [results[a]["avg_length"] for a in approaches]
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(10, 15))
    
    # Plot success rates
    axes[0].bar(approaches, success_rates, color='green')
    axes[0].set_title('Success Rate Comparison')
    axes[0].set_ylim([0, 1])
    axes[0].set_ylabel('Success Rate')
    
    # Plot average rewards
    axes[1].bar(approaches, avg_rewards, color='blue')
    axes[1].set_title('Average Reward Comparison')
    axes[1].set_ylabel('Average Reward')
    
    # Plot average episode lengths
    axes[2].bar(approaches, avg_lengths, color='red')
    axes[2].set_title('Average Episode Length Comparison')
    axes[2].set_ylabel('Steps')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'benchmark_comparison.png'))
    plt.close()
    
    print(f"Comparison plots saved to {os.path.join(output_dir, 'benchmark_comparison.png')}")

    # Also save results as text
    with open(os.path.join(output_dir, 'benchmark_results.txt'), 'w') as f:
        f.write("Benchmark Results\n")
        f.write("================\n\n")
        
        for approach in approaches:
            f.write(f"{approach}:\n")
            f.write(f"  Success Rate: {results[approach]['success_rate']:.2f}\n")
            f.write(f"  Average Reward: {results[approach]['avg_reward']:.2f}\n")
            f.write(f"  Average Episode Length: {results[approach]['avg_length']:.2f}\n")
            f.write("\n")