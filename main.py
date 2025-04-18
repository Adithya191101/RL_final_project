#!/usr/bin/env python3
"""
Main entry point for robotic arm reinforcement learning project.
Provides training and evaluation capabilities with different configurations.
"""
import os
import sys
import time
import yaml
import argparse
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Robotic Arm RL Training and Evaluation")
    
    # Mode selection
    parser.add_argument("--mode", type=str, choices=["train", "evaluate", "benchmark"], 
                       default="train", help="Operating mode (train, evaluate, benchmark)")
    
    # Configuration
    parser.add_argument("--config", type=str, default="configs/fixed_training.yaml", 
                       help="Path to configuration file")
    
    # Training settings
    parser.add_argument("--timesteps", type=int, default=200000, 
                       help="Number of timesteps for training")
    parser.add_argument("--save_dir", type=str, default="models", 
                       help="Directory to save models")
    
    # Evaluation settings
    parser.add_argument("--model_path", type=str, default=None, 
                       help="Path to model for evaluation")
    parser.add_argument("--episodes", type=int, default=10, 
                       help="Number of episodes for evaluation")
    
    # Visualization
    parser.add_argument("--render", action="store_true", 
                       help="Enable rendering")
    parser.add_argument("--camera_view", action="store_true", 
                       help="Enable camera visualization")
    
    return parser.parse_args()

def train(args):
    """Train a reinforcement learning agent."""
    print(f"Starting training with config: {args.config}")
    print(f"Timesteps: {args.timesteps}")
    
    # Import training module from src
    sys.path.append('.')
    from src.train import train as train_agent
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Set rendering based on args
    config["environment"]["render"] = args.render
    
    # Override timesteps if specified
    if args.timesteps:
        config["rl"]["total_timesteps"] = args.timesteps
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Run training
    train_agent(config, args)
    
    print(f"Training completed. Model saved to {args.save_dir}")

def evaluate(args):
    """Evaluate a trained reinforcement learning agent."""
    print(f"Starting evaluation with config: {args.config}")
    
    if not args.model_path:
        # Try to use the latest model if none specified
        models_dir = Path(args.save_dir)
        if models_dir.exists():
            models = list(models_dir.glob("*.pt"))
            if models:
                latest_model = max(models, key=lambda p: p.stat().st_mtime)
                args.model_path = str(latest_model)
                print(f"Using latest model: {args.model_path}")
            else:
                print("No models found in models directory.")
                return
        else:
            print(f"Model path not specified and {args.save_dir} does not exist.")
            return
    
    # Import evaluation module from src
    sys.path.append('.')
    from src.train import evaluate as evaluate_agent
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Set rendering based on args
    config["environment"]["render"] = args.render
    
    # Run evaluation
    evaluate_agent(config, args)

def benchmark(args):
    """Run benchmarks comparing different approaches."""
    print("Starting benchmark comparison")
    
    # Import benchmark module
    sys.path.append('.')
    try:
        from src.benchmark import run_benchmark
        
        # Run benchmark with args
        run_benchmark(args)
    except ImportError:
        print("Benchmark module not found.")

def main():
    """Main entry point."""
    args = parse_args()
    
    # Call the appropriate function based on mode
    if args.mode == "train":
        train(args)
    elif args.mode == "evaluate":
        evaluate(args)
    elif args.mode == "benchmark":
        benchmark(args)
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")