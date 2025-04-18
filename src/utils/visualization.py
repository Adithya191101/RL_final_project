"""
Visualization utilities for the robotic arm trajectory planning system.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List, Dict, Any, Optional, Tuple
import warnings

# Check if matplotlib 3D is available
MATPLOTLIB_3D_AVAILABLE = True
try:
    from mpl_toolkits.mplot3d import Axes3D
except ImportError:
    MATPLOTLIB_3D_AVAILABLE = False
    warnings.warn("Matplotlib 3D plotting is not available. Some visualization functions will be limited.")


def plot_trajectory(
    joint_positions: np.ndarray,
    ee_positions: np.ndarray,
    goal_position: np.ndarray,
    obstacle_positions: List[np.ndarray],
    fig_path: Optional[str] = None
):
    """
    Plot a trajectory in 3D.
    
    Args:
        joint_positions: Sequence of joint positions (N, num_joints)
        ee_positions: Sequence of end effector positions (N, 3)
        goal_position: Goal position (3,)
        obstacle_positions: List of obstacle positions, each (3,)
        fig_path: Path to save the figure (optional)
    """
    if not MATPLOTLIB_3D_AVAILABLE:
        # Fall back to 2D plotting if 3D is not available
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot Top View (XY)
        ax1.plot(ee_positions[:, 0], ee_positions[:, 1], "b-", label="Trajectory")
        ax1.scatter(ee_positions[0, 0], ee_positions[0, 1], c="g", marker="o", s=100, label="Start")
        ax1.scatter(ee_positions[-1, 0], ee_positions[-1, 1], c="r", marker="*", s=100, label="End")
        ax1.scatter(goal_position[0], goal_position[1], c="y", marker="*", s=200, label="Goal")
        
        for i, obs_pos in enumerate(obstacle_positions):
            ax1.scatter(obs_pos[0], obs_pos[1], c="r", marker="o", s=100, alpha=0.7, 
                       label="Obstacle" if i == 0 else "")
        
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_title("Top View (XY)")
        ax1.legend()
        ax1.grid(True)
        
        # Plot Side View (XZ)
        ax2.plot(ee_positions[:, 0], ee_positions[:, 2], "b-", label="Trajectory")
        ax2.scatter(ee_positions[0, 0], ee_positions[0, 2], c="g", marker="o", s=100, label="Start")
        ax2.scatter(ee_positions[-1, 0], ee_positions[-1, 2], c="r", marker="*", s=100, label="End")
        ax2.scatter(goal_position[0], goal_position[2], c="y", marker="*", s=200, label="Goal")
        
        for i, obs_pos in enumerate(obstacle_positions):
            ax2.scatter(obs_pos[0], obs_pos[2], c="r", marker="o", s=100, alpha=0.7)
        
        ax2.set_xlabel("X")
        ax2.set_ylabel("Z")
        ax2.set_title("Side View (XZ)")
        ax2.grid(True)
        
    else:
        # Use 3D plotting
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        
        # Plot trajectory
        ax.plot(
            ee_positions[:, 0],
            ee_positions[:, 1],
            ee_positions[:, 2],
            "b-",
            label="Trajectory"
        )
        
        # Plot start and end positions
        ax.scatter(
            ee_positions[0, 0],
            ee_positions[0, 1],
            ee_positions[0, 2],
            c="g",
            marker="o",
            s=100,
            label="Start"
        )
        ax.scatter(
            ee_positions[-1, 0],
            ee_positions[-1, 1],
            ee_positions[-1, 2],
            c="r",
            marker="*",
            s=100,
            label="End"
        )
        
        # Plot goal
        ax.scatter(
            goal_position[0],
            goal_position[1],
            goal_position[2],
            c="y",
            marker="*",
            s=200,
            label="Goal"
        )
        
        # Plot obstacles
        for i, obs_pos in enumerate(obstacle_positions):
            ax.scatter(
                obs_pos[0],
                obs_pos[1],
                obs_pos[2],
                c="r",
                marker="o",
                s=100,
                alpha=0.7,
                label="Obstacle" if i == 0 else ""
            )
        
        # Set labels and title
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Robot Arm Trajectory")
        
        # Add legend
        ax.legend()
    
    # Save or show
    if fig_path:
        plt.savefig(fig_path)
    else:
        plt.show()
    
    plt.close()


def create_trajectory_animation(
    joint_positions: np.ndarray,
    ee_positions: np.ndarray,
    goal_position: np.ndarray,
    obstacle_positions: List[np.ndarray],
    anim_path: Optional[str] = None
):
    """
    Create an animation of the trajectory.
    
    Args:
        joint_positions: Sequence of joint positions (N, num_joints)
        ee_positions: Sequence of end effector positions (N, 3)
        goal_position: Goal position (3,)
        obstacle_positions: List of obstacle positions, each (3,)
        anim_path: Path to save the animation (optional)
    """
    if not MATPLOTLIB_3D_AVAILABLE:
        # Fall back to 2D animation if 3D is not available
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Set limits
        x_min, x_max = np.min(ee_positions[:, 0]) -.5, np.max(ee_positions[:, 0]) + .5
        y_min, y_max = np.min(ee_positions[:, 1]) - .5, np.max(ee_positions[:, 1]) + .5
        z_min, z_max = np.min(ee_positions[:, 2]) - .5, np.max(ee_positions[:, 2]) + .5
        
        ax1.set_xlim([x_min, x_max])
        ax1.set_ylim([y_min, y_max])
        ax2.set_xlim([x_min, x_max])
        ax2.set_ylim([z_min, z_max])
        
        # Set labels and titles
        ax1.set_xlabel("X")
        ax1.set_ylabel("Y")
        ax1.set_title("Top View (XY)")
        ax1.grid(True)
        
        ax2.set_xlabel("X")
        ax2.set_ylabel("Z")
        ax2.set_title("Side View (XZ)")
        ax2.grid(True)
        
        # Plot goal
        ax1.scatter(goal_position[0], goal_position[1], c="y", marker="*", s=200, label="Goal")
        ax2.scatter(goal_position[0], goal_position[2], c="y", marker="*", s=200, label="Goal")
        
        # Plot obstacles
        for i, obs_pos in enumerate(obstacle_positions):
            ax1.scatter(obs_pos[0], obs_pos[1], c="r", marker="o", s=100, alpha=0.7, 
                       label="Obstacle" if i == 0 else "")
            ax2.scatter(obs_pos[0], obs_pos[2], c="r", marker="o", s=100, alpha=0.7)
        
        # Initialize plots
        line1, = ax1.plot([], [], "b-", label="Trajectory")
        point1, = ax1.plot([], [], "bo", markersize=8, label="End Effector")
        line2, = ax2.plot([], [], "b-")
        point2, = ax2.plot([], [], "bo", markersize=8)
        
        ax1.legend()
        
        def init():
            line1.set_data([], [])
            point1.set_data([], [])
            line2.set_data([], [])
            point2.set_data([], [])
            return line1, point1, line2, point2
        
        def animate(i):
            line1.set_data(ee_positions[:i+1, 0], ee_positions[:i+1, 1])
            point1.set_data([ee_positions[i, 0]], [ee_positions[i, 1]])
            line2.set_data(ee_positions[:i+1, 0], ee_positions[:i+1, 2])
            point2.set_data([ee_positions[i, 0]], [ee_positions[i, 2]])
            return line1, point1, line2, point2
        
    else:
        # Use 3D animation
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        
        # Set labels and title
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title("Robot Arm Trajectory Animation")
        
        # Set axis limits
        x_min, x_max = np.min(ee_positions[:, 0]) -.5, np.max(ee_positions[:, 0]) + .5
        y_min, y_max = np.min(ee_positions[:, 1]) - .5, np.max(ee_positions[:, 1]) + .5
        z_min, z_max = np.min(ee_positions[:, 2]) - .5, np.max(ee_positions[:, 2]) + .5
        
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.set_zlim([z_min, z_max])
        
        # Initialize plots
        line, = ax.plot([], [], [], "b-", label="Trajectory")
        point, = ax.plot([], [], [], "bo", markersize=8, label="End Effector")
        
        # Plot goal
        ax.scatter(
            goal_position[0],
            goal_position[1],
            goal_position[2],
            c="y",
            marker="*",
            s=200,
            label="Goal"
        )
        
        # Plot obstacles
        for i, obs_pos in enumerate(obstacle_positions):
            ax.scatter(
                obs_pos[0],
                obs_pos[1],
                obs_pos[2],
                c="r",
                marker="o",
                s=100,
                alpha=0.7,
                label="Obstacle" if i == 0 else ""
            )
        
        # Add legend
        ax.legend()
        
        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
            return line, point
        
        def animate(i):
            line.set_data(ee_positions[:i+1, 0], ee_positions[:i+1, 1])
            line.set_3d_properties(ee_positions[:i+1, 2])
            point.set_data([ee_positions[i, 0]], [ee_positions[i, 1]])
            point.set_3d_properties([ee_positions[i, 2]])
            return line, point
    
    # Create animation
    anim = FuncAnimation(
        fig, animate, init_func=init,
        frames=len(ee_positions), interval=50, blit=True
    )
    
    # Save or show
    if anim_path:
        anim.save(anim_path, writer="ffmpeg")
    else:
        plt.show()
    
    plt.close()


def plot_reward_history(rewards: List[float], fig_path: Optional[str] = None):
    """
    Plot the reward history.
    
    Args:
        rewards: List of rewards
        fig_path: Path to save the figure (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Reward History")
    plt.grid(True)
    
    # Add moving average
    window_size = min(100, len(rewards))
    if window_size > 0:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(np.arange(window_size-1, len(rewards)), moving_avg, 'r-', label=f'Moving Avg (window={window_size})')
        plt.legend()
    
    # Save or show
    if fig_path:
        plt.savefig(fig_path)
    else:
        plt.show()
    
    plt.close()


def visualize_controller_switching(
    ee_positions: np.ndarray,
    controller_types: List[str],
    obstacle_positions: List[np.ndarray],
    goal_position: np.ndarray,
    fig_path: Optional[str] = None
):
    """
    Visualize controller switching along a trajectory.
    
    Args:
        ee_positions: Sequence of end effector positions (N, 3)
        controller_types: List of controller types for each step ('rl' or 'mpc')
        obstacle_positions: List of obstacle positions, each (3,)
        goal_position: Goal position (3,)
        fig_path: Path to save the figure (optional)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    
    # Separate trajectory by controller type
    rl_indices = [i for i, c in enumerate(controller_types) if c == 'rl']
    mpc_indices = [i for i, c in enumerate(controller_types) if c == 'mpc']
    
    # Plot RL trajectory segments
    for i in range(len(rl_indices)-1):
        if rl_indices[i+1] == rl_indices[i] + 1:  # consecutive indices
            ax.plot(
                ee_positions[rl_indices[i]:rl_indices[i+1]+1, 0],
                ee_positions[rl_indices[i]:rl_indices[i+1]+1, 1],
                ee_positions[rl_indices[i]:rl_indices[i+1]+1, 2],
                "b-",
                label="RL" if i == 0 else ""
            )
    
    # Plot MPC trajectory segments
    for i in range(len(mpc_indices)-1):
        if mpc_indices[i+1] == mpc_indices[i] + 1:  # consecutive indices
            ax.plot(
                ee_positions[mpc_indices[i]:mpc_indices[i+1]+1, 0],
                ee_positions[mpc_indices[i]:mpc_indices[i+1]+1, 1],
                ee_positions[mpc_indices[i]:mpc_indices[i+1]+1, 2],
                "g-",
                label="MPC" if i == 0 else ""
            )
    
    # Plot start and end positions
    ax.scatter(
        ee_positions[0, 0],
        ee_positions[0, 1],
        ee_positions[0, 2],
        c="g",
        marker="o",
        s=100,
        label="Start"
    )
    ax.scatter(
        ee_positions[-1, 0],
        ee_positions[-1, 1],
        ee_positions[-1, 2],
        c="r",
        marker="*",
        s=100,
        label="End"
    )
    
    # Plot goal
    ax.scatter(
        goal_position[0],
        goal_position[1],
        goal_position[2],
        c="y",
        marker="*",
        s=200,
        label="Goal"
    )
    
    # Plot obstacles
    for i, obs_pos in enumerate(obstacle_positions):
        ax.scatter(
            obs_pos[0],
            obs_pos[1],
            obs_pos[2],
            c="r",
            marker="o",
            s=100,
            alpha=0.7,
            label="Obstacle" if i == 0 else ""
        )
    
    # Set labels and title
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Controller Switching Visualization")
    
    # Add legend
    ax.legend()
    
    # Save or show
    if fig_path:
        plt.savefig(fig_path)
    else:
        plt.show()
    
    plt.close()