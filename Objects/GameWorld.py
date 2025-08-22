import pygame
from Objects.Map import Map
from Objects.Player import Player
from Settings import global_settings as settings

class GameWorld:
    def __init__(self, camera):
        #TODO: make it so 2 players can be initialized
        self.game_map = Map(file_location="map.txt")
        self.playerOne = Player(30, 40, self.game_map, 1)
        self.playerTwo = Player(90, 30, self.game_map, 2)
        self.camera = camera

        self.last_collision_time = 0
        # TODO: make this a setting
        self.collision_cooldown = 2000
        
        #TODO: make this random
        self.seeker = self.playerOne
        
        self.players = [self.playerOne, self.playerTwo]

    def update(self, keys):
        
        now = pygame.time.get_ticks()
        
        self.camera.mark_seeker(self.seeker)
        
        # Update the players
        for player in self.players:
            player.update(keys, self.game_map)

            if (
                self.seeker.hitbox.colliderect(player.hitbox)
                and player != self.seeker
                and now - self.last_collision_time >= self.collision_cooldown
            ):
                if settings.DEBUG_MODE:
                    print(f"Seeker {self.seeker.player_id} collided with Player {player.player_id}")

                
                # Swap seeker
                player.color = (0, 255, 0)
                self.seeker.color = (255, 0, 0)
                self.seeker = player

                
                # Reset cooldown timer
                self.last_collision_time = now


            

        # When map will have more logic, it will be updated here
        # self.map.update()