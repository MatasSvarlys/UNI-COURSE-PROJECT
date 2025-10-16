import pygame
import numpy as np
from Objects.Camera import Camera
from Objects.Map import Map
from Objects.Player import Player
from Settings import global_settings as settings

class GameWorld:
    def __init__(self):

        self.gameMap = Map(file_location="map.txt")
        # TODO: make players get generated in general area and without separate objects
        self.playerOne = Player(30, 40, 0, True)
        self.playerTwo = Player(90, 30, 1)
        self.players = [self.playerOne, self.playerTwo]
        self.camera = Camera()


        self.baseSurface = pygame.Surface((settings.WINDOW_WIDTH, settings.WINDOW_HEIGHT), pygame.SRCALPHA)

        self.surfaces = []

        # for seeker collisions
        self.last_collision_time = 0
        # TODO: make this a setting
        self.collision_cooldown = 2000
        
    def update(self, keys, flags):
        
        if flags[0]:
            self.reset()
            flags[0] = False
            return
        
        now = pygame.time.get_ticks()
        
        # Update the players
        for player in self.players:
            player.update(keys, self.gameMap.get_nearby_collision_rects(player.hitbox))

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
                
                flags[0] = True


                # Reset cooldown timer
                self.last_collision_time = now
        
        # self.camera.follow_with_offset(self.players[0].hitbox, offset_x=0, offset_y=-settings.WINDOW_HEIGHT // 4)
        self.camera.follow_between_players(self.playerOne.hitbox, self.playerTwo.hitbox)
        self.camera.manual_nudge(0, -settings.WINDOW_HEIGHT // 4)
        # When map will have more logic, it will be updated here
        # self.map.update()

    def reset(self):
        self.playerOne.set_position(30, 40)
        self.playerTwo.set_position(90, 30)
        self.playerOne.isSeeker = True
        self.playerTwo.isSeeker = False

        return
    
    def draw(self):

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
        return len(self.get_state_for_player(self.playerOne.player_id))

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
        dx = otherPlayer.position.x - player.position.x
        dy = otherPlayer.position.y - player.position.y

        state.extend([
            dx,
            dy,
            float(otherPlayer.isSeeker)
        ])

        # Since a NN won't accept a rect, use only the corner position
        # Could be nice to test how it feels using all corners or just the middle position
        for rect in self.gameMap.collision_rects:
            state.extend(rect.topleft)

        # Instead of using all values, it's possible to use LiDAR (raycasting) to only get 
        # positions of a few rectangles, or even better, just the distance to the rectangle

        return np.array(state, dtype=np.float32)