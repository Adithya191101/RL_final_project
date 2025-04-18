"""
Hybrid controller that integrates RL with MPC for robotic arm trajectory planning.
Uses RL for global planning and MPC for local refinement.
"""
import numpy as np
import gym
from typing import Dict, List, Tuple, Optional, Any, Union
import sys
import os

# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.ppo_agent import RollingHorizonPPO
from rl.smdp_agent import RollingSMDPAgent
from controllers.mpc_controller import MPCController


class HybridController:
    """Hybrid controller that integrates RL with MPC for trajectory planning."""
    
    def __init__(self, config: Dict[str, Any], state_dim: int, action_dim: int):
        """
        Initialize the hybrid controller.
        
        Args:
            config: Configuration dictionary with controller parameters
            state_dim: Dimension of the state
            action_dim: Dimension of the action
        """
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Check if SMDP is enabled
        self.use_smdp = config["environment"].get("use_smdp", False)
        
        # Check if MPC should be used
        self.use_mpc = config["environment"].get("use_mpc", True)
        
        # Initialize controllers
        if self.use_smdp:
            self.rl_controller = RollingSMDPAgent(config, state_dim, action_dim)
        else:
            self.rl_controller = RollingHorizonPPO(config, state_dim, action_dim)
            
        if self.use_mpc:
            self.mpc_controller = MPCController(config)
        else:
            # No MPC, just use RL
            self.mpc_controller = None
        
        # Switching parameters
        self.obstacle_distance_threshold = 0.2  # Switch to MPC if obstacles are close
        self.goal_distance_threshold = 0.3  # Switch to MPC if close to goal
        self.current_controller = "rl"  # Start with RL
    
    def select_controller(self, state: np.ndarray, obstacles: List[Dict[str, Any]]) -> str:
        """
        Select which controller to use based on the current state.
        
        Args:
            state: Current state
            obstacles: List of obstacle dictionaries with position and size
            
        Returns:
            Controller type ("rl" or "mpc")
        """
        # If MPC is disabled, always use RL
        if not self.use_mpc:
            return "rl"
            
        # Parse state
        num_joints = min(6, len(state) - 12)  # Handle different robot types
        joint_pos = state[:num_joints]
        ee_pos = state[num_joints:num_joints+3]
        goal_pos = state[num_joints+3:num_joints+6]
        obstacle_pos = state[num_joints+6:]
        
        # Calculate distance to goal
        goal_distance = np.linalg.norm(ee_pos - goal_pos)
        
        # Calculate minimum distance to obstacles
        min_obstacle_distance = float('inf')
        for i in range(0, len(obstacle_pos), 3):
            if i + 2 < len(obstacle_pos):
                obstacle = obstacle_pos[i:i+3]
                distance = np.linalg.norm(ee_pos - obstacle)
                min_obstacle_distance = min(min_obstacle_distance, distance)
        
        # Select controller based on distances
        if min_obstacle_distance < self.obstacle_distance_threshold:
            # Close to obstacle, use MPC for precise avoidance
            return "mpc"
        elif goal_distance < self.goal_distance_threshold:
            # Close to goal, use MPC for precise positioning
            return "mpc"
        else:
            # Otherwise use RL for global planning
            return "rl"
    
    def select_action(self, state: np.ndarray, env, obstacles: List[Dict[str, Any]] = None, deterministic: bool = False) -> np.ndarray:
        """
        Select an action using the appropriate controller.
        
        Args:
            state: Current state
            env: Environment to simulate (needed for RL rolling horizon)
            obstacles: List of obstacle dictionaries (optional)
            
        Returns:
            Action
        """
        # If MPC is disabled, always use RL
        if not self.use_mpc:
            self.current_controller = "rl"
            return self.rl_controller.select_action(state, env, deterministic=deterministic)
            
        # Parse state to extract joint positions and velocities
        num_joints = min(6, len(state) - 12)  # Handle different robot types
        joint_pos = state[:num_joints]
        
        # Get joint velocities from environment
        joint_vel = np.zeros_like(joint_pos)
        for i, joint_idx in enumerate(env.joint_indices):
            if i < num_joints:  # Safety check
                joint_state = env.p.getJointState(env.robot_id, joint_idx)
                joint_vel[i] = joint_state[1]
        
        # Extract goal position and obstacle information
        goal_pos = state[num_joints+3:num_joints+6]
        
        # Select controller
        controller_type = self.select_controller(state, obstacles or [])
        self.current_controller = controller_type
        
        if controller_type == "rl" or not self.use_mpc:
            # Use RL controller
            return self.rl_controller.select_action(state, env, deterministic=deterministic)
        else:
            # Use MPC controller
            # First, we need to convert the goal position to target joint positions
            # In a real implementation, this would involve inverse kinematics
            # For simplicity, we'll assume we have a function to do this
            target_joint_pos = self._inverse_kinematics(goal_pos, joint_pos, env)
            
            # Get MPC action
            action = self.mpc_controller.optimize_trajectory(
                joint_pos, joint_vel, target_joint_pos, obstacles
            )
            
            # Scale action to match RL action range
            scaled_action = np.clip(action, -1.0, 1.0)
            
            return scaled_action
    
    def update_with_real_experience(self, state: np.ndarray, action: Union[np.ndarray, Dict], 
                               reward: float, next_state: np.ndarray, done: bool, info: Dict = None):
        """
        Update the RL agent with real experience.
        
        Args:
            state: Current state
            action: Taken action
            reward: Received reward
            next_state: Next state
            done: Done flag
            info: Info dictionary (optional)
        """
        # Update RL controller regardless of which controller generated the action
        if self.use_smdp:
            self.rl_controller.update_with_real_experience(state, action, reward, next_state, done, info)
        else:
            self.rl_controller.update_with_real_experience(state, action, reward, next_state, done)
    
    def _inverse_kinematics(self, target_ee_pos: np.ndarray, current_joint_pos: np.ndarray, env) -> np.ndarray:
        """
        Perform inverse kinematics to convert end effector position to joint positions.
        This is a simplified version - a real implementation would use a proper IK solver.
        
        Args:
            target_ee_pos: Target end effector position
            current_joint_pos: Current joint positions
            env: Environment (for PyBullet IK)
            
        Returns:
            Target joint positions
        """
        # Use PyBullet IK if available
        if hasattr(env, 'p'):
            # Get the end effector link index
            ee_link = env.end_effector_index
            
            # Calculate IK
            joint_positions = env.p.calculateInverseKinematics(
                env.robot_id,
                ee_link,
                target_ee_pos,
                maxNumIterations=100,
                residualThreshold=1e-4
            )
            
            # Return joint positions for the actuated joints
            return np.array(joint_positions[:6])
        else:
            # Fallback to a simple approximation if PyBullet IK is not available
            # This is just a placeholder - not a real IK solution
            return current_joint_pos
    
    def update(self):
        """
        Update the RL policy after collecting experience.
        Delegates to the underlying RL controller.
        """
        # Call update on the RL controller if it has the method
        if hasattr(self.rl_controller, 'update'):
            self.rl_controller.update()
    
    def save_model(self, path: str):
        """
        Save the RL model.
        
        Args:
            path: Path to save the model
        """
        self.rl_controller.save_model(path)
    
    def load_model(self, path: str):
        """
        Load a saved RL model.
        
        Args:
            path: Path to the saved model
        """
        self.rl_controller.load_model(path)