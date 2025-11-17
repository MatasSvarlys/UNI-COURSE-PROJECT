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
        self.gameMap = Map(file_location="map.txt")
        # TODO: make players get generated in general area and without separate objects
        self.playerOne = Player(map_settings.PLAYER_ONE_START[0], map_settings.PLAYER_ONE_START[1], 0, True)
        
        self.playerTwo = Player(map_settings.PLAYER_TWO_START[0], map_settings.PLAYER_TWO_START[1], 1)
        self.players = [self.playerOne, self.playerTwo]

        if not rl_settings.TRAINING_MODE:
            self.camera = Camera()
            self.baseSurface = pygame.Surface((settings.WINDOW_WIDTH, settings.WINDOW_HEIGHT), pygame.SRCALPHA)

            self.surfaces = []

        # for seeker collisions
        self.last_collision_time = 0
        # TODO: make this a setting
        self.collision_cooldown = 2000


        self.previous_positions = {
            self.playerOne.player_id: self.playerOne.position.copy(),
            self.playerTwo.player_id: self.playerTwo.position.copy()
        }
    
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

            if self.distanceBetween <= 100:
                player.reward += rl_settings.REWARD_FOR_PROXIMITY

            if self.is_player_out_of_bounds(player):
                player.reward -= rl_settings.PENALTY_FOR_RUNNING_OUT_THE_MAP


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
        self.playerOne.set_position(map_settings.PLAYER_ONE_START[0], map_settings.PLAYER_ONE_START[1])
        self.playerTwo.set_position(np.random.randint(map_settings.TILE_SIZE*2, (map_settings.MAP_WIDTH-2)*map_settings.TILE_SIZE), map_settings.PLAYER_TWO_START[1])
        self.playerOne.isSeeker = True
        self.playerTwo.isSeeker = False

        return
    
    def draw(self):
        print("a")
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
        self.surfaces.append(players_surface)

        self.camera.draw_surfaces(self.surfaces)

    def get_state_array_size(self):
        return len(self.get_state_for_player(self.playerOne.player_id)[0])

    def get_state_for_player(self, playerID):
        
        if playerID == self.playerOne.player_id:
            player = self.playerOne
            otherPlayer = self.playerTwo
        else:
            player = self.playerTwo
            otherPlayer = self.playerOne

        state = []

        # Player own state (6 values)
        state.extend([
            player.position.x,
            player.position.y,
            player.movementVector.x,
            player.movementVector.y,
            float(player.grounded),  
            float(player.isSeeker)
        ])

        # Other player relative state (4 values)
        
        # An argument could be made to use direct position instead of relative
        # dx = otherPlayer.position.x - player.position.x
        # dy = otherPlayer.position.y - player.position.y

        state.extend([
            otherPlayer.position.x,
            otherPlayer.position.y,
        ])

        # Since a NN won't accept a rect, use only the corner position
        # Could be nice to test how it feels using all corners or just the middle position
        for rect in self.gameMap.collision_rects:
            state.extend(rect.topleft)

        # Instead of using all values, it's possible to use LiDAR (raycasting) to only get 
        # positions of a few rectangles, or even better, just the distance to the rectangle

        reward = player.reward

        return (np.array(state, dtype=np.float32), reward)