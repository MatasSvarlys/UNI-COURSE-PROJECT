import os
import random
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
        self.map_files = self.get_map_files()
        self.current_map_file = random.choice(self.map_files)
        
        self.gameMap = Map(file_location=self.current_map_file)
        
        # TODO: make players get generated in general area and without separate objects
        self.playerOne = Player(self.gameMap.p1StartPos[0], self.gameMap.p1StartPos[1], 0, True)
        
        self.playerTwo = Player(self.gameMap.p2StartPos[0], self.gameMap.p2StartPos[1], 1)
        self.players = [self.playerOne, self.playerTwo]

        if not rl_settings.TRAINING_MODE:
            self.camera = Camera()
            self.baseSurface = pygame.Surface((settings.WINDOW_WIDTH, settings.WINDOW_HEIGHT), pygame.SRCALPHA)

            self.surfaces = []

        # for seeker collisions
        self.last_collision_time = 0
        # TODO: make this a setting
        self.collision_cooldown = 2000

        self.lidar_num_rays = 32
        self.lidar_ray_angles = [i * 360.0 / self.lidar_num_rays for i in range(self.lidar_num_rays)]

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
    
    def load_random_map(self):
        self.current_map_file = random.choice(self.map_files)
        self.gameMap = Map(file_location=self.current_map_file)
        
        # Reset player positions based on new map
        self.playerOne.set_position(self.gameMap.p1StartPos[0], self.gameMap.p1StartPos[1])
        self.playerTwo.set_position(self.gameMap.p2StartPos[0], self.gameMap.p2StartPos[1])

    def distance_between_two_rects(self, pos1, pos2):
        return ((pos1.x - pos2.x) ** 2 + (pos1.y - pos2.y) ** 2) ** 0.5

    def is_player_out_of_bounds(self, player):
        map_width_pixels = map_settings.MAP_WIDTH * map_settings.TILE_SIZE
        
        # TODO: make it a setting
        buffer_zone = map_settings.TILE_SIZE * 1.5
        
        return (player.position.x < buffer_zone or 
                player.position.x > map_width_pixels - buffer_zone)

    def update(self, keys):
        if states.isTerminated:
            self.reset()
            states.isTerminated = False
            states.startNewEpisode()
            return
        
        now = pygame.time.get_ticks()
        self.distanceBetween = self.distance_between_two_rects(self.playerOne.position, self.playerTwo.position)

        # Update the players
        for player in self.players:
            player.update(keys, self.gameMap.get_nearby_collision_rects(player.hitbox))

            if self.distanceBetween <= 50:
                player.reward += rl_settings.REWARD_FOR_PROXIMITY

            # if self.is_player_out_of_bounds(player):
            #     player.reward -= rl_settings.PENALTY_FOR_RUNNING_INTO_WALL


            if (
                self.players[0].hitbox.colliderect(self.players[1].hitbox)
                and (self.players[0].isSeeker or self.players[1].isSeeker)
                # and now - self.last_collision_time >= self.collision_cooldown
            ):
                if settings.DEBUG_MODE:
                    print(f"Seeker collided with Player")
                
                # Swap seeker
                # self.players[0].isSeeker = not self.players[0].isSeeker
                # self.players[1].isSeeker = not self.players[1].isSeeker
                

                self.players[0].collided_with_seeker()
                self.players[1].collided_with_seeker()

                states.isTerminated = True

                # Reset cooldown timer
                self.last_collision_time = now
        

        if not rl_settings.TRAINING_MODE:
            # self.camera.follow_with_offset(self.players[0].hitbox, offset_x=0, offset_y=-settings.WINDOW_HEIGHT // 4)
            self.camera.follow_between_players(self.playerOne.hitbox, self.playerTwo.hitbox)
            self.camera.manual_nudge(0, -settings.WINDOW_HEIGHT // 4)


        # When map will have more logic, it will be updated here
        # self.map.update()

    def reset(self):
        self.load_random_map()

        self.playerOne.isSeeker = True
        self.playerTwo.isSeeker = False

        return
    
    def draw_lidar_rays(self, surface, player_position):
        
        
        for angle in self.lidar_ray_angles:
            _, collision_point = self.cast_lidar_ray(player_position, angle)
            
            if collision_point:
                # Draw ray up to collision point (red line)
                pygame.draw.line(surface, (255, 0, 0), player_position, collision_point, 1)
                # Draw small circle at collision point
                pygame.draw.circle(surface, (255, 0, 0), (int(collision_point[0]), int(collision_point[1])), 3)
            else:
                # Draw full ray if no collision (green line)
                angle_rad = np.radians(angle)
                end_x = player_position.x + 1000 * np.cos(angle_rad)
                end_y = player_position.y + 1000 * np.sin(angle_rad)
                pygame.draw.line(surface, (0, 255, 0), player_position, (end_x, end_y), 1)

    def draw(self):
        if rl_settings.TRAINING_MODE:
            return
        # Reset the surfaces
        self.surfaces = []

        # Draw map
        map_surface = self.baseSurface.copy()
        map_surface = self.gameMap.draw_to_surface(map_surface)
        self.surfaces.append(map_surface)

        # Draw players
        players_surface = self.baseSurface.copy()
        for player in self.players:
            players_surface = player.draw_to_surface(players_surface)
            if player.isSeeker:
                self.draw_lidar_rays(players_surface, pygame.math.Vector2(player.position.x + settings.PLAYER_WIDTH/2, player.position.y + settings.PLAYER_HEIGHT/2))
        self.surfaces.append(players_surface)

        self.camera.draw_surfaces(self.surfaces)


    def cast_lidar_ray(self, start_pos, angle, max_distance=1000):
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



    def get_state_array_size(self):
        return len(self.get_state_for_player(self.playerOne.player_id)[0])

    def get_state_for_player(self, playerID):
        
        if playerID == self.playerOne.player_id:
            player = self.playerOne
            otherPlayer = self.playerTwo
        else:
            player = self.playerTwo
            otherPlayer = self.playerOne

        map_width = map_settings.MAP_WIDTH * map_settings.TILE_SIZE
        map_height = map_settings.MAP_HEIGHT * map_settings.TILE_SIZE
        state = []

        # Player own state (6 values)
        state.extend([
            player.position.x / map_width,
            player.position.y / map_height,
            player.movementVector.x / settings.PLAYER_MAX_SPEED,
            player.movementVector.y / settings.PLAYER_MAX_FSPEED,
            float(player.grounded),  
            float(player.isSeeker)
        ])

        # Other player relative state (4 values)
        
        # An argument could be made to use direct position instead of relative
        # dx = otherPlayer.position.x - player.position.x
        # dy = otherPlayer.position.y - player.position.y

        state.extend([
            otherPlayer.position.x / map_width,
            otherPlayer.position.y / map_height,
            otherPlayer.movementVector.x / settings.PLAYER_MAX_SPEED,
            otherPlayer.movementVector.y / settings.PLAYER_MAX_FSPEED,
        ])

        # Since a NN won't accept a rect, use only the corner position
        # Could be nice to test how it feels using all corners or just the middle position
        # for rect in self.gameMap.collision_rects:
        #     state.extend(rect.topleft)


        # Instead of using all values, it's possible to use LiDAR (raycasting) to only get 
        # positions of a few rectangles, or even better, just the distance to the rectangle
        
        # TODO: use the middle of the player instead of the top right
        lidar_readings = self.get_lidar_readings(pygame.math.Vector2(player.position.x + settings.PLAYER_WIDTH/2, player.position.y + settings.PLAYER_HEIGHT/2))
        normalized_lidar = [dist / 1000.0 for dist in lidar_readings]  # max_distance=1000
        # print(f"{lidar_readings}\n")

        state.extend(normalized_lidar)

        
        reward = player.reward

        return (np.array(state, dtype=np.float32), reward)