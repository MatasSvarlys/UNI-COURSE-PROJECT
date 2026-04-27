from concurrent.futures import ThreadPoolExecutor
import os
import random
import cv2
from PIL import Image
from numba import njit
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
        self.executor = ThreadPoolExecutor(max_workers=len(self.rlPlayers))

        # Surfaces for drawing
        self.baseSurface = pygame.Surface((settings.WINDOW_WIDTH, settings.WINDOW_HEIGHT), pygame.SRCALPHA)
        self.scaled_ai_view = pygame.Surface((rl_settings.IMAGE_WIDTH, rl_settings.IMAGE_HEIGHT))
        self.players_surface = pygame.Surface((settings.WINDOW_WIDTH, settings.WINDOW_HEIGHT), pygame.SRCALPHA)
        self.lidar_surface = pygame.Surface((settings.WINDOW_WIDTH, settings.WINDOW_HEIGHT), pygame.SRCALPHA)
        self.last_lidar_results = [None] * len(self.players)

        self.captureOccured = False

        if not settings.HEADLESS_MODE:
            self.camera = Camera()

        self.lidar_num_rays = rl_settings.LIDAR_RAY_COUNT
        self.lidar_ray_angles = [i * 360.0 / self.lidar_num_rays for i in range(self.lidar_num_rays)]
        self.lidar_ray_angles_rad = np.radians(self.lidar_ray_angles)
        self.dir_x = np.cos(self.lidar_ray_angles_rad)
        self.dir_y = np.sin(self.lidar_ray_angles_rad)

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

        p1_res, p2_res = self.gameMap.get_random_spawns()
        
        # Reset player positions based on new map
        self.playerOne.set_position(p1_res[0], p1_res[1])
        self.playerTwo.set_position(p2_res[0], p2_res[1])

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
        
    def draw_lidar_rays(self, surface, player_idx):
        result = self.last_lidar_results[player_idx]
        if result is None:
            return
        
        for i in range(self.lidar_num_rays):
            end = (result['end_x'][i], result['end_y'][i])
            if result['hit_wall'][i]:
                pygame.draw.line(surface, (255, 0, 0, 50), result['start'], end, 1)
                pygame.draw.circle(surface, (255, 0, 0), (int(end[0]), int(end[1])), 1)
            else:
                pygame.draw.line(surface, (0, 255, 0), result['start'], end, 1)
    
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
        for i, player in enumerate(self.players):
            if i not in self.rlPlayers:
                self.update_discovery(i)
            self.draw_lidar_rays(self.lidar_surface, i)
        self.surfaces.append(self.lidar_surface)

        self.camera.draw_surfaces(self.surfaces)


    def cast_lidar_ray_and_update(self, start_pos, memory, other_player, player_idx):

        start_gx = start_pos.x / map_settings.TILE_SIZE
        start_gy = start_pos.y / map_settings.TILE_SIZE
        max_grid_dist = rl_settings.LIDAR_MAX_DISTANCE / map_settings.TILE_SIZE
                
        wall_dist_grid, hit_wall = numba_raycast(
                start_gx, start_gy, 
                self.dir_x, self.dir_y,
                max_grid_dist, 
                self.gameMap.blockGrid, 
                memory, 
                self.lidar_map_colors[0], 
                self.lidar_map_colors[1]
            )
        
        dist_world = wall_dist_grid * map_settings.TILE_SIZE
        end_x_arr = start_pos.x + self.dir_x * dist_world
        end_y_arr = start_pos.y + self.dir_y * dist_world

        player_seen = rl_settings.TOGGLE_VISIBLE_PLAYERS_IN_OBSERVATION
        if not player_seen:
            for i in range(self.lidar_num_rays):
                if other_player.hitbox.clipline(start_pos, (end_x_arr[i], end_y_arr[i])):
                    player_seen = True
                    break
        
        self.last_lidar_results[player_idx] = {
            'start': start_pos,
            'end_x': end_x_arr,
            'end_y': end_y_arr,
            'hit_wall': hit_wall,
            'dist': dist_world
        }

        return player_seen
    
    def get_all_agent_observations(self):
        # Create tasks only for agents controlled by RL
        futures = {
            idx: self.executor.submit(self.get_player_observation, idx)
            for idx in self.rlPlayers
        }
        
        # Collect results as they finish
        agent_names = list(rl_settings.RL_CONTROL.keys())
        results = {agent_names[idx]: future.result() for idx, future in futures.items()}
        return results

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
        p_center = pygame.math.Vector2(
            self.players[player_idx].position.x + settings.PLAYER_WIDTH / 2,
            self.players[player_idx].position.y + settings.PLAYER_HEIGHT / 2
        )
        memory = self.player_memories[player_idx]
        other_player = self.players[1 - player_idx]
        
        other_visible = self.cast_lidar_ray_and_update(
            p_center, memory, other_player, player_idx
        )

        other_player.is_visible_to_current = other_visible

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
        
        # Image.fromarray(obs_np.astype(np.uint8)).save(f"map_debug_{player_idx}.png")
        return obs_np
    
@njit
def numba_raycast(start_gx, start_gy, dir_x, dir_y, max_grid_dist, block_grid, memory, wall_color, floor_color):
    num_rays = len(dir_x)
    grid_h, grid_w = block_grid.shape
    
    # Results to pass back
    wall_dist = np.empty(num_rays)
    end_x_arr = np.empty(num_rays)
    end_y_arr = np.empty(num_rays)
    hit_wall = np.zeros(num_rays, dtype=np.bool_)

    for i in range(num_rays):
        dx, dy = dir_x[i], dir_y[i]
        map_x, map_y = int(start_gx), int(start_gy)

        delta_x = abs(1.0 / dx) if dx != 0 else 1e30
        delta_y = abs(1.0 / dy) if dy != 0 else 1e30

        step_x = 1 if dx >= 0 else -1
        step_y = 1 if dy >= 0 else -1

        side_x = (map_x + 1.0 - start_gx) * delta_x if dx >= 0 else (start_gx - map_x) * delta_x
        side_y = (map_y + 1.0 - start_gy) * delta_y if dy >= 0 else (start_gy - map_y) * delta_y

        dist = 0.0
        hit = False

        # THE HEAVY LIFTING: Compiled by Numba
        while dist < max_grid_dist:
            if side_x < side_y:
                dist = side_x
                side_x += delta_x
                map_x += step_x
            else:
                dist = side_y
                side_y += delta_y
                map_y += step_y

            if map_x < 0 or map_x >= grid_w or map_y < 0 or map_y >= grid_h:
                break

            if block_grid[map_y, map_x] == 1:
                memory[map_y, map_x] = wall_color
                hit = True
                break
            else:
                memory[map_y, map_x] = floor_color

        dist = min(dist, max_grid_dist)
        wall_dist[i] = dist
        # We return these to update the class later
        hit_wall[i] = hit

    return wall_dist, hit_wall
