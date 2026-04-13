import os
import random
from PIL import Image
import pygame
import numpy as np
import Objects.States as states
from Objects.Camera import Camera
from Objects.Map import Map
from Objects.Player import Player
from Settings import global_settings as settings
from Settings import map_settings
from Settings import rl_settings

class GameWorld:
    def __init__(self):
        self._preload_maps()
        
        self.current_map_file = random.choice(self.map_files)
        self.gameMap = self.map_cache[self.current_map_file]
        
        # TODO: make players get generated in general area and without separate objects
        self.playerOne = Player(self.gameMap.p1StartPos[0], self.gameMap.p1StartPos[1], 0, True)
        
        self.playerTwo = Player(self.gameMap.p2StartPos[0], self.gameMap.p2StartPos[1], 1)
        self.players = [self.playerOne, self.playerTwo]

        self.rlPlayers = [i for i, (_, v) in enumerate(rl_settings.RL_CONTROL.items()) if v]

        # Surfaces for drawing
        self.baseSurface = pygame.Surface((settings.WINDOW_WIDTH, settings.WINDOW_HEIGHT), pygame.SRCALPHA)
        self.scaled_ai_view = pygame.Surface((rl_settings.IMAGE_WIDTH, rl_settings.IMAGE_HEIGHT))
        self.players_surface = pygame.Surface((settings.WINDOW_WIDTH, settings.WINDOW_HEIGHT), pygame.SRCALPHA)
        self.lidar_surface = pygame.Surface((settings.WINDOW_WIDTH, settings.WINDOW_HEIGHT), pygame.SRCALPHA)
        
        
        if not settings.HEADLESS_MODE:
            self.camera = Camera()

        self.lidar_num_rays = rl_settings.LIDAR_RAY_COUNT
        self.lidar_ray_angles = [i * 360.0 / self.lidar_num_rays for i in range(self.lidar_num_rays)]


        # TODO: keep only the memory of players that are being trained
        self.player_memories = [
            np.full((self.gameMap.grid_height, self.gameMap.grid_width), 127, dtype=np.uint8) 
            for _ in range(len(self.players))
        ]

        self.previous_positions = {
            self.playerOne.player_id: self.playerOne.position.copy(),
            self.playerTwo.player_id: self.playerTwo.position.copy()
        }
    
    def get_map_files(self):
        maps_dir = "maps"        
        map_files = []

        for file in os.listdir(maps_dir):
            if file.endswith(".txt") and file.startswith("map"):
                map_files.append(os.path.join(maps_dir, file))
        
        return map_files
    
    def _preload_maps(self):
        maps_dir = "maps"
        self.map_files = []
        
        for file in os.listdir(maps_dir):
            if file.endswith(".txt") and file.startswith("map"):
                file_path = os.path.join(maps_dir, file)
                self.map_files.append(file_path)
        
        # Load all maps into cache
        print(f"Preloading {len(self.map_files)} maps...")
        self.map_cache = {}
        for map_file in self.map_files:
            self.map_cache[map_file] = Map(file_location=map_file)
        print(f"Preloaded {len(self.map_cache)} maps successfully.")
    

    def load_random_map(self):
        self.current_map_file = random.choice(self.map_files)
        # Get preloaded map instead of creating new one
        self.gameMap = self.map_cache[self.current_map_file]
        
        # Reset player positions based on new map
        self.playerOne.set_position(self.gameMap.p1StartPos[0], self.gameMap.p1StartPos[1])
        self.playerTwo.set_position(self.gameMap.p2StartPos[0], self.gameMap.p2StartPos[1])

    def players_collided(self):
        for player_index in self.rlPlayers:
            player = self.players[player_index]
            if player.isSeeker:
                player.reward += rl_settings.REWARD_FOR_WINNING
            else:
                player.reward -= rl_settings.REWARD_FOR_WINNING

    def update(self, playerActions):

        # Update the players
        for player in self.players:
            player.reward = 0
            player.update(playerActions[player.player_id], self.gameMap.get_nearby_collision_rects(player.hitbox))
            
        if not states.isTerminated and self.players[0].hitbox.colliderect(self.players[1].hitbox):
                self.players_collided()
                states.isTerminated = True
                if settings.DEBUG_MODE:
                    print(f"Seeker collided with Player")

        # Give reward for existing
        for playerID in self.rlPlayers:
            player = self.players[playerID]
            if player.isSeeker:
                player.reward += -rl_settings.REWARD_FOR_EXISTING
            else:
                player.reward += rl_settings.REWARD_FOR_EXISTING
        
        if not settings.HEADLESS_MODE:
            self.camera.manual_nudge(0, 0)

        # When map will have more logic, it will be updated here
        # self.map.update()


    def reset(self):
        self.load_random_map()

        self.player_memories = [
            np.full((self.gameMap.grid_height, self.gameMap.grid_width), 127, dtype=np.uint8) 
            for _ in range(len(self.players))
        ]

        self.playerOne.isSeeker = True
        self.playerTwo.isSeeker = False
        
        return
    
    def draw_lidar_rays(self, surface, player_position):
        
        
        for angle in self.lidar_ray_angles:
            _, collision_point = self.cast_lidar_ray(player_position, angle)
            
            if collision_point:
                # Draw ray up to collision point (red line)
                pygame.draw.line(surface, (255, 0, 0, 50), player_position, collision_point, 1)
                # Draw small circle at collision point
                pygame.draw.circle(surface, (255, 0, 0), (int(collision_point[0]), int(collision_point[1])), 1)
            else:
                # Draw full ray if no collision (green line)
                angle_rad = np.radians(angle)
                end_x = player_position.x + rl_settings.LIDAR_MAX_DISTANCE * np.cos(angle_rad)
                end_y = player_position.y + rl_settings.LIDAR_MAX_DISTANCE * np.sin(angle_rad)
                pygame.draw.line(surface, (0, 255, 0), player_position, (end_x, end_y), 1)

    def draw(self):

        self.surfaces = []

        # Draw map
        self.gameMap.drawMapOntoSurface(self.baseSurface)
        self.surfaces.append(self.baseSurface)

        # Draw players
        self.players_surface.fill((0, 0, 0, 0))
        for player in self.players:
            player.draw_to_surface(self.players_surface)
        self.surfaces.append(self.players_surface)
        
        
        self.lidar_surface.fill((0, 0, 0, 0))
        for player in self.players:
            center = pygame.math.Vector2(
                player.position.x + settings.PLAYER_WIDTH/2, 
                player.position.y + settings.PLAYER_HEIGHT/2
            )
            self.draw_lidar_rays(self.lidar_surface, center)
        self.surfaces.append(self.lidar_surface)

        self.camera.draw_surfaces(self.surfaces)


    def cast_lidar_ray(self, start_pos, angle, max_distance=rl_settings.LIDAR_MAX_DISTANCE):
        # Convert angle to radians
        angle_rad = np.radians(angle)
        
        # Calculate end point of the ray
        end_x = start_pos.x + max_distance * np.cos(angle_rad)
        end_y = start_pos.y + max_distance * np.sin(angle_rad)
        
        # to see the lowest distance, use one of those 12th grade algorithms
        min_distance = max_distance
        collision_point = None

        # TODO: get collisions to the other player too
        for rect in self.gameMap.collision_rects:
            if self.line_rect_intersection(start_pos, (end_x, end_y), rect):
                # Find the exact collision point
                clipped_line = rect.clipline(start_pos, (end_x, end_y))
                if clipped_line:
                    # Get the closest point of intersection
                    point1, point2 = clipped_line
                    d1 = start_pos.distance_to(point1)
                    d2 = start_pos.distance_to(point2)
                    
                    if d1 < min_distance:
                        min_distance = d1
                        collision_point = point1
                    if d2 < min_distance:
                        min_distance = d2
                        collision_point = point2
        
        # Return both distance and collision point for drawing
        return min_distance, collision_point
    
    def line_rect_intersection(self, startPos, endPos, rect):
        return bool(rect.clipline(startPos, endPos))

    def get_lidar_readings(self, player_position):
        readings = []
        
        for angle in self.lidar_ray_angles:
            distance, _ = self.cast_lidar_ray(player_position, angle)
            readings.append(distance)
        
        return readings


    def get_state_screenshot(self, width=rl_settings.IMAGE_WIDTH, height=rl_settings.IMAGE_HEIGHT):

        self.gameMap.drawMapOntoSurface(self.baseSurface)

        for player in self.players:
            player.draw_to_surface(self.baseSurface)
            
        pygame.transform.scale(self.baseSurface, (width, height), self.scaled_ai_view)        
        
        img_array = pygame.surfarray.array3d(self.scaled_ai_view)
        
        # Image.fromarray(img_array.astype(np.uint8)).save("debug_state.png")
        
        img_array = img_array.transpose(2, 1, 0)
        
        return img_array.astype(np.uint8)

    def get_reward(self, id):
        return self.players[id].reward
    

    def trace_and_update_memory(self, player_idx, start_pos, end_pos, hit_occured):
        memory = self.player_memories[player_idx]
        
        # Convert world coordinates to grid coordinates
        x0, y0 = self.gameMap.world_to_grid_coordinates(start_pos.x, start_pos.y)
        x1, y1 = self.gameMap.world_to_grid_coordinates(end_pos.x, end_pos.y)

        # Get the distance of game blocks from the player to the collision point 
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        
        # The total steps to reach the end point
        n = 1 + dx + dy
        
        # Decide on which direction to step to
        x_inc = 1 if x1 > x0 else -1
        y_inc = 1 if y1 > y0 else -1
        
        # Decide the next steps direction
        error = dx - dy
        dx *= 2
        dy *= 2

        for _ in range(n):
            # Bounds check
            if 0 <= x < self.gameMap.grid_width and 0 <= y < self.gameMap.grid_height:
                # If this is the last tile and we actually hit a wall
                if x == x1 and y == y1 and hit_occured:
                    memory[y, x] = 0 # Mark as Wall
                    break 
                elif memory[y, x] != 0:
                    # If it wasnt marked as a wall before, mark it as passable
                    memory[y, x] = 255
            
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx

    def update_discovery(self, player_idx):
        # Center of player for ray casting
        p_center = pygame.math.Vector2(
            self.players[player_idx].position.x + settings.PLAYER_WIDTH / 2,
            self.players[player_idx].position.y + settings.PLAYER_HEIGHT / 2
        )

        for angle in self.lidar_ray_angles:
            _, collision_point = self.cast_lidar_ray(p_center, angle)
            
            if collision_point:
                # Cast the point to a Vector2 for the tracer
                end_point = pygame.math.Vector2(collision_point[0], collision_point[1])
                self.trace_and_update_memory(player_idx, p_center, end_point, True)
            else:
                # No hit: trace to max distance
                angle_rad = np.radians(angle)
                end_x = p_center.x + rl_settings.LIDAR_MAX_DISTANCE * np.cos(angle_rad)
                end_y = p_center.y + rl_settings.LIDAR_MAX_DISTANCE * np.sin(angle_rad)
                self.trace_and_update_memory(player_idx, p_center, pygame.math.Vector2(end_x, end_y), False)

    def get_player_observation(self, player_idx):
        self.update_discovery(player_idx)
        
        # Get the permanent memory of the map
        obs = self.player_memories[player_idx].copy()
        
        # Overlay players: 
        # We use a unique value so the AI knows "This is a player, not a wall"
        for i, p in enumerate(self.players):
            gx, gy = self.gameMap.world_to_grid_coordinates(p.position.x, p.position.y)
            if 0 <= gx < self.gameMap.grid_width and 0 <= gy < self.gameMap.grid_height:
                # Value 50 for self, 200 for opponent (high contrast)
                obs[gy, gx] = 50 if i == player_idx else 200
                
        # self.save_memory_as_image(player_idx)
        # Return normalized (1, H, W) for PyTorch/Tensorflow
        return obs.astype(np.float32) / 255.0
    
    def save_memory_as_image(self, player_idx, filename="map_debug.png"):
        # 1. Grab the 2D array (H, W)
        grid = self.player_memories[player_idx]
        
        # 2. (Optional) Let's draw the players on top so we know where they are
        # We create a copy so we don't permanently draw players into the "memory"
        visual_grid = grid.copy()
        
        for i, p in enumerate(self.players):
            gx, gy = self.gameMap.world_to_grid_coordinates(p.position.x, p.position.y)
            if 0 <= gx < self.gameMap.grid_width and 0 <= gy < self.gameMap.grid_height:
                # Mark self as a different gray/white value to stand out
                visual_grid[gy, gx] = 50 if i == player_idx else 200
        
        # 3. Convert the NumPy array to a PIL Image object and save
        img = Image.fromarray(visual_grid)
        img.save(filename)
        print(f"Saved memory map for player {player_idx} to {filename}")