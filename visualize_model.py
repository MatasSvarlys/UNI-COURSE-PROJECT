import numpy as np
import pygame
import torch
from Settings import rl_settings, global_settings as settings, map_settings
from Objects.DQNAgent import DQNAgent
from Objects.Map import Map
import os


class DQNVisualizer:

    
    def __init__(self, agent, game_map, grid_resolution):

        self.agent = agent
        self.game_map = game_map
        
        # Map dimensions
        self.map_width = map_settings.MAP_WIDTH * map_settings.TILE_SIZE
        self.map_height = map_settings.MAP_HEIGHT * map_settings.TILE_SIZE
        
        # Calculate grid resolution based on map size and scale factor
        self.grid_resolution_x = map_settings.MAP_WIDTH * grid_resolution
        self.grid_resolution_y = map_settings.MAP_HEIGHT * grid_resolution
        
        # Lidar settings (match GameWorld)
        self.lidar_num_rays = 32
        self.lidar_ray_angles = [i * 360.0 / self.lidar_num_rays for i in range(self.lidar_num_rays)]
        
        # Color scheme for actions (RGB)
        self.action_colors = {
            0: (100, 100, 100),    # NOOP - Gray
            1: (165, 42, 42),      # LEFT - Brown
            2: (255, 100, 0),      # RIGHT - Orange
            3: (0, 255, 100),      # JUMP - Green
            4: (100, 0, 255),      # LEFT_JUMP - Purple
            5: (255, 255, 0),      # RIGHT_JUMP - Yellow
            6: (0, 0, 0),          # IN_BLOCK - Black
        }
        
        # Pre-compute grid positions
        self.grid_x = np.linspace(0, self.map_width - 1, self.grid_resolution_x)
        self.grid_y = np.linspace(0, self.map_height - 1, self.grid_resolution_y)
        self.cell_width = self.map_width / self.grid_resolution_x
        self.cell_height = self.map_height / self.grid_resolution_y
        
        # Cache for heatmap (updated when needed)
        self.cached_heatmap = None
        self.cache_params = None
        
    def cast_lidar_ray(self, start_pos, angle, max_distance=1000):
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
    
    def is_position_in_block(self, pos):
        # Create a rect representing the player at this position
        player_rect = pygame.Rect(
            int(pos[0]),
            int(pos[1]),
            settings.PLAYER_WIDTH/2,
            settings.PLAYER_HEIGHT/2
        )
        
        # Check collision with any solid block
        for collision_rect in self.game_map.collision_rects:
            if player_rect.colliderect(collision_rect):
                return True
        return False
    
    def get_lidar_readings(self, player_pos):
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
        # Check if we can use cached version
        cache_key = (p2_pos, p1_vel, p2_vel)
        if self.cache_params == cache_key and self.cached_heatmap is not None:
            return self.cached_heatmap
        
        heatmap = np.zeros((self.grid_resolution_y, self.grid_resolution_x), dtype=int)
        
        for i, x in enumerate(self.grid_x):
            for j, y in enumerate(self.grid_y):
                p1_pos = (x, y)
                
                # Check if this position would put the player in a block
                if self.is_position_in_block(p1_pos):
                    heatmap[j, i] = 6  # IN_BLOCK action
                else:
                    action_idx, _ = self.get_action_and_qvalues(p1_pos, p2_pos, p1_vel, p2_vel)
                    heatmap[j, i] = action_idx
        
        self.cached_heatmap = heatmap
        self.cache_params = cache_key
        return heatmap
    
    def draw_to_surface(self, surface, p2_pos, p1_vel=(0, 0), p2_vel=(0, 0), alpha=128):

        heatmap = self.generate_heatmap(p2_pos, p1_vel, p2_vel)
        
        # Create a temporary surface for the heatmap with alpha channel
        heatmap_surface = pygame.Surface((self.map_width, self.map_height), pygame.SRCALPHA)
        
        for i in range(self.grid_resolution_x):
            for j in range(self.grid_resolution_y):
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
                
                # Draw all cells (including IN_BLOCK)
                pygame.draw.rect(heatmap_surface, color_with_alpha, rect)
        
        # Blit the heatmap surface onto the main surface
        surface.blit(heatmap_surface, (0, 0))
        
        return surface
    
    def draw_legend(self, surface, x=10, y=10):
        font = pygame.font.Font(None, 24)
        
        action_names = {
            0: rl_settings.ACTIONS[0],
            1: rl_settings.ACTIONS[1],
            2: rl_settings.ACTIONS[2],
            3: rl_settings.ACTIONS[3],
            4: rl_settings.ACTIONS[4],
            5: rl_settings.ACTIONS[5],
            6: "IN_BLOCK"
        }
        
        for i, (action_idx, color) in enumerate(self.action_colors.items()):
            action_name = action_names[action_idx]
            
            # Draw color box
            box_rect = pygame.Rect(x, y + i * 30, 20, 20)
            pygame.draw.rect(surface, color, box_rect)
            pygame.draw.rect(surface, (255, 255, 255), box_rect, 1)  # Border
            
            # Draw action name
            text = font.render(action_name, True, (255, 255, 255))
            surface.blit(text, (x + 30, y + i * 30 + 2))


def load_trained_agent():
    state_size = 42  
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


if __name__ == "__main__":
    agent = load_trained_agent()
    
    # Load a map
    map_files = [f for f in os.listdir("maps") if f.endswith(".txt") and f.startswith("map")]
    for idx, map in enumerate(map_files):
        print(f"{idx}: {map}")
    

    game_map = Map(file_location=os.path.join("maps", map_files[13]))
    
    # Create visualizer
    visualizer = DQNVisualizer(agent, game_map, grid_resolution=10)
    
    # Initialize pygame
    pygame.init()
    
    # Calculate screen size to fit map + legend on the right
    legend_width = 200
    screen_width = map_settings.MAP_WIDTH * map_settings.TILE_SIZE + legend_width
    screen_height = map_settings.MAP_HEIGHT * map_settings.TILE_SIZE
    
    screen = pygame.display.set_mode((screen_width, screen_height))
    
    # Use player 2 start position
    p2_pos = game_map.p2StartPos
    p1_vel = (0, 0)
    p2_vel = (0, 0)
    
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        

        # move the target if needed
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
        pygame.draw.rect(screen, (255, 0, 0), 
                pygame.Rect(int(p2_pos[0] + settings.PLAYER_WIDTH/2 - 10), 
                       int(p2_pos[1] + settings.PLAYER_HEIGHT/2 - 10), 
                       20, 20))
        
        # Draw legend on the right side of the map
        legend_x = map_settings.MAP_WIDTH * map_settings.TILE_SIZE + 10
        legend_y = 10
        visualizer.draw_legend(screen, x=legend_x, y=legend_y)
        
        pygame.display.flip()
    
    pygame.quit()