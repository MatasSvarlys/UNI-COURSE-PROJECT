import os
import random
import cv2
from PIL import Image
import pygame
import numpy as np
from sympy import rad
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
        
        self.captureOccured = False

        if not settings.HEADLESS_MODE:
            self.camera = Camera()

        self.lidar_num_rays = rl_settings.LIDAR_RAY_COUNT
        self.lidar_ray_angles = [i * 360.0 / self.lidar_num_rays for i in range(self.lidar_num_rays)]
        self.lidar_map_colors = [100, 200, 150] # wall, floor, unknown

        # TODO: keep only the memory of players that are being trained
        self.player_memories = [
            np.full((self.gameMap.grid_height, self.gameMap.grid_width), 128, dtype=np.uint8) 
            for _ in range(len(self.players))
        ]
        self.scale_x = rl_settings.IMAGE_WIDTH / settings.WINDOW_WIDTH
        self.scale_y = rl_settings.IMAGE_HEIGHT / settings.WINDOW_HEIGHT

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
            
        if self.players[0].hitbox.colliderect(self.players[1].hitbox):
                self.players_collided()
                self.captureOccured = True

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
            np.full((self.gameMap.grid_height, self.gameMap.grid_width), self.lidar_map_colors[2], dtype=np.uint8) 
            for _ in range(len(self.players))
        ]

        self.playerOne.isSeeker = True
        self.playerTwo.isSeeker = False
        
        return
    
    def draw_lidar_rays(self, surface, player_position, player_idx):
        
        
        for angle in self.lidar_ray_angles:
            _, collision_point, _ = self.cast_lidar_ray_and_update(player_position, angle, self.player_memories[player_idx], player_idx)
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
            self.draw_lidar_rays(self.lidar_surface, center, player.player_id)
        self.surfaces.append(self.lidar_surface)

        self.camera.draw_surfaces(self.surfaces)


    def cast_lidar_ray_and_update(self, start_pos, angle, memory, player_idx, max_distance=rl_settings.LIDAR_MAX_DISTANCE):
        # Convert angle to radians
        rad = np.radians(angle)
        
        dir_x = np.cos(rad)
        dir_y = np.sin(rad)
        
        # Start grid coords
        map_x = int(start_pos.x / map_settings.TILE_SIZE)
        map_y = int(start_pos.y / map_settings.TILE_SIZE)
        
        delta_dist_x = abs(1 / dir_x) if dir_x != 0 else 1e30
        delta_dist_y = abs(1 / dir_y) if dir_y != 0 else 1e30


        step_x = 1 if dir_x >= 0 else -1
        side_dist_x = (map_x + 1.0 - start_pos.x / map_settings.TILE_SIZE) * delta_dist_x if dir_x >= 0 else (start_pos.x / map_settings.TILE_SIZE - map_x) * delta_dist_x
        
        step_y = 1 if dir_y >= 0 else -1
        side_dist_y = (map_y + 1.0 - start_pos.y / map_settings.TILE_SIZE) * delta_dist_y if dir_y >= 0 else (start_pos.y / map_settings.TILE_SIZE - map_y) * delta_dist_y
        
        max_grid_dist = max_distance / map_settings.TILE_SIZE
        current_grid_dist = 0.0
        hit_wall = False

        # walk through the grid until we hit a wall or exceed max distance
        while current_grid_dist < max_grid_dist:
            if side_dist_x < side_dist_y:
                current_grid_dist = side_dist_x
                side_dist_x += delta_dist_x
                map_x += step_x
            else:
                current_grid_dist = side_dist_y
                side_dist_y += delta_dist_y
                map_y += step_y
            
            # mark the memory along the way
            if 0 <= map_x < self.gameMap.grid_width and 0 <= map_y < self.gameMap.grid_height:
                if self.gameMap.blockGrid[map_y][map_x] == 1:
                    memory[map_y, map_x] = self.lidar_map_colors[0] # Mark wall
                    hit_wall = True
                    break
                else:
                    memory[map_y, map_x] = self.lidar_map_colors[1] # Mark floor
            else:
                break

        wall_world_dist = min(current_grid_dist, max_grid_dist) * map_settings.TILE_SIZE
        ray_end_world = (start_pos.x + dir_x * wall_world_dist, 
                         start_pos.y + dir_y * wall_world_dist)
        
        other_player = self.players[1 - player_idx]
        # clipline is fast enough to run once per ray
        player_seen = bool(other_player.hitbox.clipline(start_pos, ray_end_world))

        # Return both distance and collision point for drawing
        return wall_world_dist, ray_end_world, player_seen    
    

    def get_state_screenshot(self, width=rl_settings.IMAGE_WIDTH, height=rl_settings.IMAGE_HEIGHT):

        self.gameMap.drawMapOntoSurface(self.baseSurface)

        colors = [(255, 255, 255), (0, 0, 0)]
        for i, player in enumerate(self.players):
            # Temporarily draw a simple rect or colored sprite
            pygame.draw.rect(self.baseSurface, colors[i], player.hitbox)
        
        pygame.transform.scale(self.baseSurface, (width, height), self.scaled_ai_view)        
        
        img_array = pygame.surfarray.array3d(self.scaled_ai_view)
        
        gray_img = img_array.mean(axis=2)
        
        # Image.fromarray(img_array.astype(np.uint8)).save("debug_state.png")
        
        gray_img = gray_img.transpose(1, 0)
        
        return gray_img.astype(np.uint8)

    def get_reward(self, id):
        return self.players[id].reward
    
    def update_discovery(self, player_idx):
        # Center of player for ray casting
        p_center = pygame.math.Vector2(
            self.players[player_idx].position.x + settings.PLAYER_WIDTH / 2,
            self.players[player_idx].position.y + settings.PLAYER_HEIGHT / 2
        )

        memory = self.player_memories[player_idx]
        other_visible = False
        
        for angle in self.lidar_ray_angles:
            # This single call handles the grid marking and player detection
            _, _, player_seen = self.cast_lidar_ray_and_update(
                p_center, angle, memory, player_idx
            )
            if player_seen:
                other_visible = True
        self.players[1 - player_idx].is_visible_to_current = other_visible

    def get_player_observation(self, player_idx):
        # Update the memory
        self.update_discovery(player_idx)
        
        obs_np = cv2.resize(self.player_memories[player_idx], 
                        (rl_settings.IMAGE_WIDTH, rl_settings.IMAGE_HEIGHT), 
                        interpolation=cv2.INTER_NEAREST)
        
        # draw the players as white and black 
        colors = [255, 0]
        for i, p in enumerate(self.players):
            is_me = (i == player_idx)
            is_visible = p.is_visible_to_current
            
            if is_me or is_visible or rl_settings.TOGGLE_VISIBLE_PLAYERS_IN_OBSERVATION:
                x1 = int(p.hitbox.x * self.scale_x)
                y1 = int(p.hitbox.y * self.scale_y)
                x2 = int((p.hitbox.x + p.hitbox.width) * self.scale_x)
                y2 = int((p.hitbox.y + p.hitbox.height) * self.scale_y)
                # Clamp to image bounds
                ix1, iy1 = int(round(x1)), int(round(y1))
                ix2, iy2 = int(round(x2)), int(round(y2))
                obs_np[max(0, iy1):min(84, iy2), max(0, ix1):min(84, ix2)] = colors[i]
        
        # Image.fromarray(obs_np.astype(np.uint8)).save("map_debug.png")
        return obs_np