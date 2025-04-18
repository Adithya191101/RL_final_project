"""
SMDP-based RL agent implementation for robotic arm trajectory planning.
Extends PPO to work with Semi-Markov Decision Processes with variable action durations.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal, Categorical
from typing import Dict, List, Tuple, Optional, Any, Union

from .ppo_agent import ActorNetwork, CriticNetwork, PPOAgent


class SMDPActorNetwork(nn.Module):
    """Actor network for SMDP that outputs both joint velocities and action duration."""
    
    def __init__(self, state_dim: int, action_dim: int, duration_dim: int, hidden_dim: int = 256):
        """
        Initialize the SMDP actor network.
        
        Args:
            state_dim: Dimension of the state
            action_dim: Dimension of the action (joint velocities)
            duration_dim: Number of possible action durations
            hidden_dim: Dimension of the hidden layers
        """
        super(SMDPActorNetwork, self).__init__()
        
        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        # Duration head
        self.duration_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, duration_dim)
        )
        
        # Action covariance (diagonal)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.distributions.Distribution, torch.Tensor, torch.distributions.Distribution]:
        """
        Forward pass through the actor network.
        
        Args:
            state: State tensor
            
        Returns:
            Tuple of (action mean, action distribution, duration logits, duration distribution)
        """
        # Extract features
        features = self.feature_extractor(state)
        
        # Get action mean and distribution
        action_mean = self.action_head(features)
        action_std = torch.exp(self.log_std)
        cov_mat = torch.diag(action_std.pow(2))
        action_dist = MultivariateNormal(action_mean, cov_mat)
        
        # Get duration logits and distribution
        duration_logits = self.duration_head(features)
        duration_dist = Categorical(logits=duration_logits)
        
        return action_mean, action_dist, duration_logits, duration_dist


class SMDPAgent:
    """SMDP agent with variable action durations for robotic arm trajectory planning."""
    
    def __init__(self, config: Dict[str, Any], state_dim: int, action_dim: int):
        """
        Initialize the SMDP agent.
        
        Args:
            config: Configuration dictionary with SMDP parameters
            state_dim: Dimension of the state
            action_dim: Dimension of the action (joint velocities)
        """
        self.config = config
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # SMDP parameters
        self.min_duration = config["environment"].get("min_action_duration", 1)
        self.max_duration = config["environment"].get("max_action_duration", 10)
        self.duration_dim = self.max_duration - self.min_duration + 1
        
        # PPO parameters
        self.gamma = config["rl"]["gamma"]
        self.clip_param = 0.2
        self.ppo_epochs = config["rl"]["n_epochs"]
        self.batch_size = config["rl"]["batch_size"]
        self.learning_rate = config["rl"]["learning_rate"]
        
        # Initialize networks
        self.actor = SMDPActorNetwork(state_dim, action_dim, self.duration_dim)
        self.critic = CriticNetwork(state_dim)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        
        # Device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
            
        # Move models to device
        self.actor.to(self.device)
        self.critic.to(self.device)
        
        if torch.cuda.is_available():
            # Use mixed precision for faster training on NVIDIA GPUs
            from torch.cuda.amp import autocast, GradScaler
            self.scaler = GradScaler()
            self.use_amp = True
            
            # Optimize CUDA performance
            torch.backends.cudnn.benchmark = True
        else:
            self.use_amp = False
        
        # Memory for experience
        self.states = []
        self.actions = []
        self.durations = []
        self.action_log_probs = []
        self.duration_log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        # Macro-action selection (option to use predefined macro-actions)
        self.use_macro_actions = config["environment"].get("use_macro_actions", False)
        self.macro_action_dim = 3  # Number of predefined macro-actions
    
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Dict[str, np.ndarray]:
        """
        Select an action and its duration given a state.
        
        Args:
            state: Current state
            
        Returns:
            Dict with 'joint_velocities' and 'duration'
        """
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).to(self.device)
        
        # Forward pass through actor
        with torch.no_grad():
            if self.use_macro_actions:
                # If using macro-actions, select from predefined set
                value = self.critic(state_tensor)
                macro_action_probs = torch.softmax(torch.randn(self.macro_action_dim), dim=0)
                macro_action_dist = Categorical(probs=macro_action_probs)
                macro_action = macro_action_dist.sample()
                log_prob = macro_action_dist.log_prob(macro_action)
                
                # Store experience for macro-action
                self.states.append(state)
                self.actions.append(macro_action.item())  # Macro-action index
                self.durations.append(0)  # Duration is determined by the environment for macro-actions
                self.action_log_probs.append(log_prob.cpu().numpy())
                self.duration_log_probs.append(0)  # No explicit duration selection
                self.values.append(value.cpu().numpy())
                
                return macro_action.item()  # Return the macro-action index
            else:
                # Standard SMDP with joint velocities and duration
                action_mean, action_dist, _, duration_dist = self.actor(state_tensor)
                action = action_dist.sample()
                duration = duration_dist.sample()
                action_log_prob = action_dist.log_prob(action)
                duration_log_prob = duration_dist.log_prob(duration)
                value = self.critic(state_tensor)
        
        # Store experience
        self.states.append(state)
        self.actions.append(action.cpu().numpy())
        self.durations.append(duration.item())
        self.action_log_probs.append(action_log_prob.cpu().numpy())
        self.duration_log_probs.append(duration_log_prob.cpu().numpy())
        self.values.append(value.cpu().numpy())
        
        # Convert to environment-expected format
        return {
            'joint_velocities': action.cpu().numpy(),
            'duration': duration.item()
        }
    
    def update(self):
        """Update the policy and value networks."""
        # Convert stored experiences to tensors
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        
        if self.use_macro_actions:
            # Update for macro-actions (similar to standard PPO)
            actions = torch.LongTensor(np.array(self.actions)).to(self.device)
            old_log_probs = torch.FloatTensor(np.array(self.action_log_probs)).to(self.device)
            
            # Compute returns and advantages
            returns, advantages = self._compute_returns_and_advantages()
            returns = torch.FloatTensor(returns).to(self.device)
            advantages = torch.FloatTensor(advantages).to(self.device)
            
            # Mini-batch training
            batch_size = min(self.batch_size, len(self.states))
            num_samples = len(self.states)
            indices = np.arange(num_samples)
            
            for _ in range(self.ppo_epochs):
                np.random.shuffle(indices)
                
                for start_idx in range(0, num_samples, batch_size):
                    end_idx = min(start_idx + batch_size, num_samples)
                    batch_indices = indices[start_idx:end_idx]
                    
                    # Get mini-batch
                    state_batch = states[batch_indices]
                    action_batch = actions[batch_indices]
                    old_log_prob_batch = old_log_probs[batch_indices]
                    return_batch = returns[batch_indices]
                    advantage_batch = advantages[batch_indices]
                    
                    # Forward pass through critic
                    value_preds = self.critic(state_batch).squeeze(-1)
                    
                    # Critic loss
                    critic_loss = nn.MSELoss()(value_preds, return_batch)
                    
                    # Update critic
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()
                    
                    # Skip actor update for macro-actions (it's predefined)
        
        else:
            # Update for standard SMDP with joint velocities and duration
            actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
            durations = torch.LongTensor(np.array(self.durations)).to(self.device)
            old_action_log_probs = torch.FloatTensor(np.array(self.action_log_probs)).to(self.device)
            old_duration_log_probs = torch.FloatTensor(np.array(self.duration_log_probs)).to(self.device)
            
            # Compute returns and advantages
            returns, advantages = self._compute_returns_and_advantages()
            returns = torch.FloatTensor(returns).to(self.device)
            advantages = torch.FloatTensor(advantages).to(self.device)
            
            # Mini-batch training
            batch_size = min(self.batch_size, len(self.states))
            num_samples = len(self.states)
            indices = np.arange(num_samples)
            
            for _ in range(self.ppo_epochs):
                np.random.shuffle(indices)
                
                for start_idx in range(0, num_samples, batch_size):
                    end_idx = min(start_idx + batch_size, num_samples)
                    batch_indices = indices[start_idx:end_idx]
                    
                    # Get mini-batch
                    state_batch = states[batch_indices]
                    action_batch = actions[batch_indices]
                    duration_batch = durations[batch_indices]
                    old_action_log_prob_batch = old_action_log_probs[batch_indices]
                    old_duration_log_prob_batch = old_duration_log_probs[batch_indices]
                    return_batch = returns[batch_indices]
                    advantage_batch = advantages[batch_indices]
                    
                    # Forward pass through networks
                    _, action_dist, _, duration_dist = self.actor(state_batch)
                    value_preds = self.critic(state_batch).squeeze(-1)
                    
                    # Get new log probs
                    new_action_log_probs = action_dist.log_prob(action_batch)
                    new_duration_log_probs = duration_dist.log_prob(duration_batch)
                    
                    # Combined log probs (action and duration)
                    old_log_probs = old_action_log_prob_batch + old_duration_log_prob_batch
                    new_log_probs = new_action_log_probs + new_duration_log_probs
                    
                    # Calculate ratios and surrogate objectives
                    ratios = torch.exp(new_log_probs - old_log_probs)
                    surr1 = ratios * advantage_batch
                    surr2 = torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param) * advantage_batch
                    
                    # PPO losses
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = nn.MSELoss()(value_preds, return_batch)
                    
                    # Update networks
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()
                    
                    self.critic_optimizer.zero_grad()
                    critic_loss.backward()
                    self.critic_optimizer.step()
        
        # Clear memory
        self._clear_memory()
    
    def _compute_returns_and_advantages(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute returns and advantages for the stored experiences.
        
        Returns:
            Tuple of (returns, advantages) as numpy arrays
        """
        # Convert to numpy arrays
        rewards = np.array(self.rewards)
        values = np.array(self.values).flatten()
        dones = np.array(self.dones)
        
        # Get durations for time-aware discounting
        if self.use_macro_actions:
            # For macro-actions, use the actual durations from info
            durations = np.array(self.durations)
        else:
            # For standard SMDP, use the selected durations
            durations = np.array([self.min_duration + d for d in self.durations])
        
        # Initialize returns and advantages
        returns = np.zeros_like(rewards)
        advantages = np.zeros_like(rewards)
        
        # Compute GAE (Generalized Advantage Estimation) with time-aware discounting
        last_gae_lambda = 0
        gae_lambda = 0.95
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0.0  # Assuming last step
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]
            
            # Time-aware discount factor based on duration
            time_discount = self.gamma ** durations[t]
            
            delta = rewards[t] + time_discount * next_value * next_non_terminal - values[t]
            last_gae_lambda = delta + time_discount * gae_lambda * next_non_terminal * last_gae_lambda
            advantages[t] = last_gae_lambda
        
        # Compute returns
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def _clear_memory(self):
        """Clear the experience memory."""
        self.states = []
        self.actions = []
        self.durations = []
        self.action_log_probs = []
        self.duration_log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def store_reward(self, reward: float, done: bool, duration: int = None):
        """
        Store a reward, done flag, and actual duration.
        
        Args:
            reward: Reward value
            done: Done flag
            duration: Actual action duration (for macro-actions)
        """
        self.rewards.append(reward)
        self.dones.append(done)
        
        # Update duration with actual value if provided (for macro-actions)
        if duration is not None and self.use_macro_actions:
            if len(self.durations) > 0:
                self.durations[-1] = duration
    
    def save_model(self, path: str):
        """
        Save the model.
        
        Args:
            path: Path to save the model
        """
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
    
    def load_model(self, path: str):
        """
        Load a saved model.
        
        Args:
            path: Path to the saved model
        """
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        
class RollingSMDPAgent:
    """SMDP agent with rolling horizon for robotic arm trajectory planning."""
    
    def __init__(self, config: Dict[str, Any], state_dim: int, action_dim: int):
        """
        Initialize the Rolling SMDP agent.
        
        Args:
            config: Configuration dictionary with SMDP parameters
            state_dim: Dimension of the state
            action_dim: Dimension of the action
        """
        self.smdp_agent = SMDPAgent(config, state_dim, action_dim)
        
        # Get rolling horizon parameters from config or use defaults
        self.horizon = config["rl"].get("horizon", 15)  # Planning horizon
        self.replan_freq = config["rl"].get("replan_freq", 5)  # Replan every N steps
        
        self.step_counter = 0
        self.current_plan = None
        self.current_plan_idx = 0
        
        # Configuration
        self.use_macro_actions = config["environment"].get("use_macro_actions", False)
    
    def plan(self, state: np.ndarray, env) -> List[Union[Dict[str, np.ndarray], int]]:
        """
        Plan a sequence of actions for the given state.
        
        Args:
            state: Current state
            env: Environment to simulate
            
        Returns:
            List of planned actions (dict or int depending on action type)
        """
        # Initialize
        planned_actions = []
        current_state = state.copy()
        
        try:
            # Simulate forward for horizon steps
            horizon_steps = self.horizon
            
            for step in range(horizon_steps):
                # Select action
                action = self.smdp_agent.select_action(current_state, deterministic=False)
                planned_actions.append(action)
                
                # Simulate one step
                next_state, reward, done, info = env.step(action)
                
                # Store reward and actual duration
                if self.use_macro_actions:
                    self.smdp_agent.store_reward(reward, done, info.get("action_duration", 1))
                else:
                    self.smdp_agent.store_reward(reward, done)
                
                # Update current state
                current_state = next_state
                
                
                if done:
                    break
            
        except Exception as e:
            # If we hit an error but have some actions, return them
            if planned_actions:
                return planned_actions
            
            # If no actions planned yet, return a default action
            if self.use_macro_actions:
                return [0]  # Default macro-action index
            else:
                return [{'joint_velocities': np.zeros(6), 'duration': 0}]  # Default SMDP action
        
        # Ensure we always return at least one action
        if not planned_actions:
            if self.use_macro_actions:
                return [0]  # Default macro-action index
            else:
                return [{'joint_velocities': np.zeros(6), 'duration': 0}]  # Default SMDP action
                
        return planned_actions
    
    def select_action(self, state: np.ndarray, env, deterministic: bool = False) -> Union[Dict[str, np.ndarray], int]:
        """
        Select an action using the rolling horizon strategy.
        
        Args:
            state: Current state
            env: Environment to simulate
            
        Returns:
            Action (dict or int depending on action type)
        """
        # Check if we need to replan
        if (self.current_plan is None or 
            self.current_plan_idx >= len(self.current_plan) or 
            self.step_counter % self.replan_freq == 0):
            
            try:
                # Create a copy of the environment for simulation
                import copy
                sim_env = copy.deepcopy(env)
                
                # Plan new sequence of actions
                self.current_plan = self.plan(state, sim_env)
                self.current_plan_idx = 0
                
                # Update networks based on simulated experience
                self.smdp_agent.update()
            except Exception as e:
                # If planning fails, just use the basic action selection
                return self.smdp_agent.select_action(state, deterministic=deterministic)
        
        # Safety check
        if not self.current_plan or self.current_plan_idx >= len(self.current_plan):
            return self.smdp_agent.select_action(state, deterministic=deterministic)
            
        # Get the next action from the current plan
        action = self.current_plan[self.current_plan_idx]
        self.current_plan_idx += 1
        self.step_counter += 1
        
        return action
    
    def update_with_real_experience(self, state: np.ndarray, action: Union[Dict[str, np.ndarray], int], 
                               reward: float, next_state: np.ndarray, done: bool, info: Dict = None):
        """
        Update the agent with real experience.
        
        Args:
            state: Current state
            action: Taken action
            reward: Received reward
            next_state: Next state
            done: Done flag
            info: Info dictionary (optional)
        """
        # Extract action components
        if self.use_macro_actions:
            # Macro-action (int)
            self.smdp_agent.states.append(state)
            self.smdp_agent.actions.append(action)
            
            # For macro-actions, duration comes from environment
            duration = info.get("action_duration", 1) if info else 1
            self.smdp_agent.durations.append(duration)
            
            # Dummy log probs (will be computed during update)
            self.smdp_agent.action_log_probs.append(0.0)
            self.smdp_agent.duration_log_probs.append(0.0)
            
            # Compute value
            state_tensor = torch.FloatTensor(state).to(self.smdp_agent.device)
            with torch.no_grad():
                value = self.smdp_agent.critic(state_tensor)
            
            # Store value, reward, and done
            self.smdp_agent.values.append(value.cpu().numpy())
            self.smdp_agent.rewards.append(reward)
            self.smdp_agent.dones.append(done)
            
        else:
            # Standard SMDP action (dict with joint_velocities and duration)
            joint_velocities = action['joint_velocities']
            duration_idx = action['duration']
            
            # Store state and action components
            self.smdp_agent.states.append(state)
            self.smdp_agent.actions.append(joint_velocities)
            self.smdp_agent.durations.append(duration_idx)
            
            # Compute log probs
            state_tensor = torch.FloatTensor(state).to(self.smdp_agent.device)
            joint_velocities_tensor = torch.FloatTensor(joint_velocities).to(self.smdp_agent.device)
            duration_tensor = torch.LongTensor([duration_idx]).to(self.smdp_agent.device)
            
            with torch.no_grad():
                _, action_dist, _, duration_dist = self.smdp_agent.actor(state_tensor)
                action_log_prob = action_dist.log_prob(joint_velocities_tensor)
                duration_log_prob = duration_dist.log_prob(duration_tensor)
                value = self.smdp_agent.critic(state_tensor)
            
            # Store log probs, value, reward, and done
            self.smdp_agent.action_log_probs.append(action_log_prob.cpu().numpy())
            self.smdp_agent.duration_log_probs.append(duration_log_prob.cpu().numpy())
            self.smdp_agent.values.append(value.cpu().numpy())
            self.smdp_agent.rewards.append(reward)
            self.smdp_agent.dones.append(done)
        
        # If episode is done, update the networks
        if done:
            self.smdp_agent.update()
    
    def save_model(self, path: str):
        """
        Save the model.
        
        Args:
            path: Path to save the model
        """
        self.smdp_agent.save_model(path)
    
    def load_model(self, path: str):
        """
        Load a saved model.
        
        Args:
            path: Path to the saved model
        """
        self.smdp_agent.load_model(path)