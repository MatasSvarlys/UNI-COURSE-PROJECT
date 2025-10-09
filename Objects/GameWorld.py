import pygame
import random
from Objects.Camera import Camera
from Objects.Map import Map
from Objects.Player import Player
from Settings import global_settings as settings

class GameWorld:
    def __init__(self):

        self.gameMap = Map(file_location="map.txt")
        self.playerOne = Player(30, 40, 1, True)
        self.playerTwo = Player(90, 30, 2)
        self.players = [self.playerOne, self.playerTwo]
        self.camera = Camera()


        self.baseSurface = pygame.Surface((settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT), pygame.SRCALPHA)

        self.surfaces = []

        # for seeker collisions
        self.last_collision_time = 0
        # TODO: make this a setting
        self.collision_cooldown = 2000
        
    def update(self, keys):
        
        now = pygame.time.get_ticks()
        
        # Update the players
        for player in self.players:
            player.update(keys, self.game_map.collision_rects)

            if (
                self.players[0].hitbox.colliderect(self.players[1].hitbox)
                and (self.players[0].isSeeker or self.players[1].isSeeker)
                and now - self.last_collision_time >= self.collision_cooldown
            ):
                if settings.DEBUG_MODE:
                    print(f"Seeker {self.seeker.player_id} collided with Player {player.player_id}")

                
                # Swap seeker
                self.players[0].isSeeker = not self.players[0].isSeeker
                self.players[1].isSeeker = not self.players[1].isSeeker
                                
                # Reset cooldown timer
                self.last_collision_time = now

        # When map will have more logic, it will be updated here
        # self.map.update()


    def draw(self):
        # Draw map
        map_surface = self.baseSurface.copy()
        self.gameMap.draw_to_surface(map_surface)

        # Draw players
        players_surface = self.baseSurface.copy()
        for player in self.players:
            player.draw_to_surface(players_surface)

        self.camera.draw_surfaces(self.surfaces)