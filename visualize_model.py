import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as patches
import torch
import os
from Settings import rl_settings
from Objects.DQNAgent import DQNAgent
from Objects.GameWorld import GameWorld
from Settings import global_settings as settings

def create_state_for_player(p1_pos, p2_pos, p1_vel, p2_vel):
    """
    Create a state array matching the format expected by the trained model.
    Based on the get_state_for_player method in GameWorld.py
    """
    # State format: [p1_x, p1_y, p1_vel_x, p1_vel_y, grounded, is_seeker, p2_x, p2_y, p2_vel_x, p2_vel_y, lidar_readings...]
    state = []

    # Player 1 own state (6 values)
    state.extend([
        p1_pos[0],  # p1_x
        p1_pos[1],  # p1_y
        p1_vel[0],  # p1_vel_x
        p1_vel[1],  # p1_vel_y
        0.0,        # grounded (assuming False for visualization)
        1.0         # is_seeker (Player 1 is seeker)
    ])

    # Player 2 relative state (4 values)
    state.extend([
        p2_pos[0],  # p2_x
        p2_pos[1],  # p2_y
        p2_vel[0],  # p2_vel_x
        p2_vel[1],  # p2_vel_y
    ])

    # Add lidar readings - for visualization we'll use placeholder values
    # In a real scenario, we'd need to generate proper lidar readings based on the map
    # But for this visualization we'll just add some reasonable placeholder values
    num_lidar_rays = 32  # From GameWorld initialization
    lidar_readings = [500.0] * num_lidar_rays  # Placeholder values
    state.extend(lidar_readings)

    return np.array(state, dtype=np.float32)

def load_trained_agent():
    """
    Load the trained DQN agent from the loads folder.
    """
    # Initialize the agent with the correct state size
    # Based on the GameWorld state format: 6 (p1 own state) + 4 (p2 state) + 32 (lidar) = 42
    state_size = 6 + 4 + 32  # 42 total features
    action_size = rl_settings.ACTION_SPACE_SIZE

    # Create agent
    agent = DQNAgent(state_size, action_size, isTraining=False)

    # Load the trained model
    model_path = "loads/player_one_policy.pth"
    if os.path.exists(model_path):
        agent.policy_network.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        print(f"Successfully loaded model from {model_path}")
    else:
        print(f"Warning: Model file {model_path} not found. Using untrained model.")

    # Set to evaluation mode
    agent.policy_network.eval()

    return agent

def get_action_for_position(agent, p1_pos, p2_pos, p1_vel, p2_vel):
    """
    Get the action the trained model would take at a given position of Player 1,
    given Player 2's position and both players' velocities.
    """
    # Create the state for the agent
    state = create_state_for_player(p1_pos, p2_pos, p1_vel, p2_vel)

    # Convert to tensor and add batch dimension
    state_tensor = torch.FloatTensor(state).unsqueeze(0)

    # Get the Q-values from the model
    with torch.no_grad():
        q_values = agent.policy_network(state_tensor)
        action_idx = torch.argmax(q_values, dim=1).item()

    return action_idx

def generate_heatmap(agent, p2_pos, p1_vel, p2_vel, grid_size=50):
    """
    Generate a heatmap showing the action the model would take at each position of Player 1,
    given Player 2's position and both players' velocities.
    """
    # Define the game space
    x_range = np.linspace(0, 600, grid_size)
    y_range = np.linspace(0, 400, grid_size)

    # Create the heatmap
    heatmap = np.zeros((grid_size, grid_size))

    for i, x in enumerate(x_range):
        for j, y in enumerate(y_range):
            # For each position of Player 1, determine the action
            p1_pos = (x, y)
            action = get_action_for_position(agent, p1_pos, p2_pos, p1_vel, p2_vel)
            heatmap[j, i] = action  # Note: j corresponds to y-axis, i to x-axis

    return heatmap, x_range, y_range

def visualize_heatmap(heatmap, x_range, y_range, p2_pos, p1_vel, p2_vel):
    """
    Visualize the heatmap with player positions and velocity vectors.
    """
    # Define colors for each action based on the ACTIONS list
    num_actions = len(rl_settings.ACTIONS)
    colors = ['white', 'blue', 'red', 'yellow', 'green', 'orange'][:num_actions]  # Different color for each action
    cmap = ListedColormap(colors)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create the heatmap
    im = ax.imshow(heatmap, extent=[x_range[0], x_range[-1], y_range[-1], y_range[0]],
                   aspect='auto', cmap=cmap, vmin=0, vmax=num_actions-1)

    # Plot Player 2 position (Player 1 positions are represented by the heatmap)
    ax.plot(p2_pos[0], p2_pos[1], 'ro', markersize=10, label='Player 2')

    # Add velocity vectors (scale for visibility)
    scale_factor = 5
    ax.arrow(p2_pos[0], p2_pos[1], p2_vel[0]*scale_factor, p2_vel[1]*scale_factor,
             head_width=10, head_length=8, fc='red', ec='red', alpha=0.7, label='P2 Velocity')
    ax.arrow(p2_pos[0]-50, p2_pos[1], p1_vel[0]*scale_factor, p1_vel[1]*scale_factor,
             head_width=10, head_length=8, fc='blue', ec='blue', alpha=0.7, label='P1 Velocity')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Action Heatmap for Player 1\nPlayer 2 at {p2_pos}, Velocities: P1={p1_vel}, P2={p2_vel}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add colorbar with action labels
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Action')
    cbar.set_ticks(range(num_actions))
    cbar.set_ticklabels(rl_settings.ACTIONS)

    plt.tight_layout()
    plt.show()

def main():
    print("Player Action Heatmap Generator")
    print(f"Available actions: {rl_settings.ACTIONS}")
    print("Enter Player 2's position and both players' velocities:")

    # Load the trained agent
    print("Loading trained model...")
    agent = load_trained_agent()

    # Get user input
    p2_x = 400
    p2_y = 200
    p1_vx = 0
    p1_vy = 0
    p2_vx = 0
    p2_vy = 0
    
    p2_pos = (p2_x, p2_y)
    p1_vel = (p1_vx, p1_vy)
    p2_vel = (p2_vx, p2_vy)

    print(f"\nGenerating heatmap for Player 2 at {p2_pos} with velocities P1={p1_vel}, P2={p2_vel}...")

    # Generate the heatmap using the trained model
    heatmap, x_range, y_range = generate_heatmap(agent, p2_pos, p1_vel, p2_vel)

    # Visualize the heatmap
    visualize_heatmap(heatmap, x_range, y_range, p2_pos, p1_vel, p2_vel)

if __name__ == "__main__":
    main()