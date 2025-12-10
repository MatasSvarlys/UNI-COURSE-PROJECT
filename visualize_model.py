import numpy as np
import pygame
import torch
from Settings import rl_settings, global_settings as settings, map_settings
from Objects.DQNAgent import DQNAgent
from Objects.Map import Map
import os


class DQNVisualizer:
    """
    Visualizes DQN Q-values as a heatmap overlay in the game world.
    Shows what action the agent prefers at each position.
    """
    
    def __init__(self, agent, game_map, grid_resolution=20):
        """
        Args:
            agent: Trained DQNAgent to visualize
            game_map: Map object containing collision rects and lidar data
            grid_resolution: Number of grid cells in each dimension (lower = faster but less detailed)
        """
        self.agent = agent
        self.game_map = game_map
        self.grid_resolution = grid_resolution
        
        # Map dimensions
        self.map_width = map_settings.MAP_WIDTH * map_settings.TILE_SIZE
        self.map_height = map_settings.MAP_HEIGHT * map_settings.TILE_SIZE
        
        # Lidar settings (match GameWorld)
        self.lidar_num_rays = 32
        self.lidar_ray_angles = [i * 360.0 / self.lidar_num_rays for i in range(self.lidar_num_rays)]
        
        # Color scheme for actions (RGB)
        self.action_colors = {
            0: (100, 100, 100),    # NOOP - Gray
            1: (0, 100, 255),      # LEFT - Blue
            2: (255, 100, 0),      # RIGHT - Orange
            3: (0, 255, 100),      # JUMP - Green
            4: (100, 0, 255),      # LEFT_JUMP - Purple
            5: (255, 255, 0),      # RIGHT_JUMP - Yellow
        }
        
        # Pre-compute grid positions
        self.grid_x = np.linspace(0, self.map_width, grid_resolution)
        self.grid_y = np.linspace(0, self.map_height, grid_resolution)
        self.cell_width = self.map_width / grid_resolution
        self.cell_height = self.map_height / grid_resolution
        
        # Cache for heatmap (updated when needed)
        self.cached_heatmap = None
        self.cache_params = None
        
    def cast_lidar_ray(self, start_pos, angle, max_distance=1000):
        """Cast a single lidar ray and return distance to nearest collision."""
        angle_rad = np.radians(angle)
        end_x = start_pos[0] + max_distance * np.cos(angle_rad)
        end_y = start_pos[1] + max_distance * np.sin(angle_rad)
        
        min_distance = max_distance
        
        for rect in self.game_map.collision_rects:
            clipped = rect.clipline(start_pos, (end_x, end_y))
            if clipped:
                point1, point2 = clipped
                d1 = np.sqrt((point1[0] - start_pos[0])**2 + (point1[1] - start_pos[1])**2)
                d2 = np.sqrt((point2[0] - start_pos[0])**2 + (point2[1] - start_pos[1])**2)
                min_distance = min(min_distance, d1, d2)
        
        return min_distance
    
    def get_lidar_readings(self, player_pos):
        """Generate lidar readings for a player position."""
        player_center = (
            player_pos[0] + settings.PLAYER_WIDTH / 2,
            player_pos[1] + settings.PLAYER_HEIGHT / 2
        )
        
        readings = []
        for angle in self.lidar_ray_angles:
            distance = self.cast_lidar_ray(player_center, angle)
            readings.append(distance / 1000.0)  # Normalize
        
        return readings
    
    def create_state(self, p1_pos, p2_pos, p1_vel=(0, 0), p2_vel=(0, 0), grounded=0.0):
        """Create state array matching the training format."""
        state = []
        
        # Player 1 state (6 values) - normalized
        state.extend([
            p1_pos[0] / self.map_width,
            p1_pos[1] / self.map_height,
            p1_vel[0] / settings.PLAYER_MAX_SPEED,
            p1_vel[1] / settings.PLAYER_MAX_FSPEED,
            float(grounded),
            1.0  # is_seeker
        ])
        
        # Player 2 state (4 values) - normalized
        state.extend([
            p2_pos[0] / self.map_width,
            p2_pos[1] / self.map_height,
            p2_vel[0] / settings.PLAYER_MAX_SPEED,
            p2_vel[1] / settings.PLAYER_MAX_FSPEED,
        ])
        
        # Lidar readings (32 values)
        lidar_readings = self.get_lidar_readings(p1_pos)
        state.extend(lidar_readings)
        
        return np.array(state, dtype=np.float32)
    
    def get_action_and_qvalues(self, p1_pos, p2_pos, p1_vel=(0, 0), p2_vel=(0, 0)):
        """Get the preferred action and all Q-values for a position."""
        state = self.create_state(p1_pos, p2_pos, p1_vel, p2_vel)
        
        # Stack frames (repeat same state for frame history)
        stacked_state = np.tile(state, rl_settings.FRAME_SKIPPING_STEPS)
        state_tensor = torch.FloatTensor(stacked_state).unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.agent.policy_network(state_tensor)
            action_idx = torch.argmax(q_values, dim=1).item()
            q_vals = q_values.cpu().numpy()[0]
        
        return action_idx, q_vals
    
    def generate_heatmap(self, p2_pos, p1_vel=(0, 0), p2_vel=(0, 0)):
        """Generate action preference heatmap across the entire map."""
        # Check if we can use cached version
        cache_key = (p2_pos, p1_vel, p2_vel)
        if self.cache_params == cache_key and self.cached_heatmap is not None:
            return self.cached_heatmap
        
        heatmap = np.zeros((self.grid_resolution, self.grid_resolution), dtype=int)
        
        for i, x in enumerate(self.grid_x):
            for j, y in enumerate(self.grid_y):
                p1_pos = (x, y)
                action_idx, _ = self.get_action_and_qvalues(p1_pos, p2_pos, p1_vel, p2_vel)
                heatmap[j, i] = action_idx
        
        self.cached_heatmap = heatmap
        self.cache_params = cache_key
        return heatmap
    
    def draw_to_surface(self, surface, p2_pos, p1_vel=(0, 0), p2_vel=(0, 0), alpha=128):
        """
        Draw the Q-value heatmap as a colored overlay on the given surface.
        
        Args:
            surface: Pygame surface to draw on
            p2_pos: Position of player 2 (the hider)
            p1_vel: Velocity of player 1 (optional)
            p2_vel: Velocity of player 2 (optional)
            alpha: Transparency (0-255, default 128 for semi-transparent)
        """
        heatmap = self.generate_heatmap(p2_pos, p1_vel, p2_vel)
        
        # Create a temporary surface for the heatmap with alpha channel
        heatmap_surface = pygame.Surface((self.map_width, self.map_height), pygame.SRCALPHA)
        
        for i in range(self.grid_resolution):
            for j in range(self.grid_resolution):
                action = heatmap[j, i]
                color = self.action_colors.get(action, (128, 128, 128))
                
                # Add alpha channel
                color_with_alpha = (*color, alpha)
                
                # Draw rectangle for this grid cell
                rect = pygame.Rect(
                    i * self.cell_width,
                    j * self.cell_height,
                    self.cell_width,
                    self.cell_height
                )
                pygame.draw.rect(heatmap_surface, color_with_alpha, rect)
        
        # Blit the heatmap surface onto the main surface
        surface.blit(heatmap_surface, (0, 0))
        
        return surface
    
    def draw_legend(self, surface, x=10, y=10):
        """Draw a legend showing what each color means."""
        font = pygame.font.Font(None, 24)
        
        for i, (action_idx, color) in enumerate(self.action_colors.items()):
            action_name = rl_settings.ACTIONS[action_idx]
            
            # Draw color box
            box_rect = pygame.Rect(x, y + i * 30, 20, 20)
            pygame.draw.rect(surface, color, box_rect)
            pygame.draw.rect(surface, (255, 255, 255), box_rect, 1)  # Border
            
            # Draw action name
            text = font.render(action_name, True, (255, 255, 255))
            surface.blit(text, (x + 30, y + i * 30 + 2))


def load_trained_agent():
    """Load a trained DQN agent from the loads folder."""
    state_size = 6 + 4 + 32  # Base state size
    action_size = rl_settings.ACTION_SPACE_SIZE
    
    agent = DQNAgent(state_size, action_size, isTraining=False)
    
    model_path = "loads/player_one_policy.pth"
    if os.path.exists(model_path):
        agent.policy_network.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu'))
        )
        print(f"Loaded model from {model_path}")
    else:
        print(f"Warning: {model_path} not found")
    
    agent.policy_network.eval()
    return agent


# Example usage for integration into GameWorld
def add_visualization_to_gameworld():
    """
    Example of how to integrate this into your GameWorld class.
    Add this to your GameWorld.__init__:
    
    if not rl_settings.TRAINING_MODE and global_settings.SHOW_Q_VISUALIZATION:
        from visualize_model_v2 import load_trained_agent, DQNVisualizer
        agent = load_trained_agent()
        self.visualizer = DQNVisualizer(agent, self.gameMap, grid_resolution=30)
    else:
        self.visualizer = None
    
    Then in your GameWorld.draw() method, before drawing players:
    
    if self.visualizer:
        # Draw Q-value heatmap as background layer
        viz_surface = self.baseSurface.copy()
        p2_pos = (self.playerTwo.position.x, self.playerTwo.position.y)
        p1_vel = (self.playerOne.movementVector.x, self.playerOne.movementVector.y)
        p2_vel = (self.playerTwo.movementVector.x, self.playerTwo.movementVector.y)
        
        viz_surface = self.visualizer.draw_to_surface(
            viz_surface, p2_pos, p1_vel, p2_vel, alpha=100
        )
        self.surfaces.append(viz_surface)
        
        # Optionally draw legend on a UI surface
        self.visualizer.draw_legend(players_surface, x=10, y=10)
    """
    pass


if __name__ == "__main__":
    # Standalone test
    print("Loading agent and map...")
    agent = load_trained_agent()
    
    # Load a map
    map_files = [f for f in os.listdir("maps") if f.endswith(".txt") and f.startswith("map")]
    for idx, map in enumerate(map_files):
        print(f"{idx}: {map}")
    if map_files:
        game_map = Map(file_location=os.path.join("maps", map_files[2]))
        print(f"Loaded map: {map_files[1]}")
    else:
        print("No maps found!")
        exit(1)
    
    # Create visualizer
    visualizer = DQNVisualizer(agent, game_map, grid_resolution=30)
    
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((
        map_settings.MAP_WIDTH * map_settings.TILE_SIZE,
        map_settings.MAP_HEIGHT * map_settings.TILE_SIZE
    ))
    pygame.display.set_caption("DQN Q-Value Visualization")
    
    # Use player 2 start position
    p2_pos = game_map.p2StartPos
    p1_vel = (0, 0)
    p2_vel = (0, 0)
    
    clock = pygame.time.Clock()
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        
        # Handle arrow keys to move p2 position
        keys = pygame.key.get_pressed()
        move_speed = 5
        if keys[pygame.K_LEFT]:
            p2_pos = (max(0, p2_pos[0] - move_speed), p2_pos[1])
        if keys[pygame.K_RIGHT]:
            p2_pos = (min(map_settings.MAP_WIDTH * map_settings.TILE_SIZE - 32, p2_pos[0] + move_speed), p2_pos[1])
        if keys[pygame.K_UP]:
            p2_pos = (p2_pos[0], max(0, p2_pos[1] - move_speed))
        if keys[pygame.K_DOWN]:
            p2_pos = (p2_pos[0], min(map_settings.MAP_HEIGHT * map_settings.TILE_SIZE - 32, p2_pos[1] + move_speed))
        
        # Clear screen
        screen.fill((0, 0, 0))
        
        # Draw map
        for rect, color in game_map.drawRects:
            pygame.draw.rect(screen, color, rect)
        
        # Draw Q-value visualization
        visualizer.draw_to_surface(screen, p2_pos, p1_vel, p2_vel, alpha=150)
        
        # Draw player 2 position
        pygame.draw.circle(screen, (255, 0, 0), 
                         (int(p2_pos[0] + settings.PLAYER_WIDTH/2), 
                          int(p2_pos[1] + settings.PLAYER_HEIGHT/2)), 
                         10)
        
        # Draw legend
        visualizer.draw_legend(screen, x=10, y=10)
        
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()