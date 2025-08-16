from Objects.Map import Map
from Objects.Player import Player
from Settings import global_settings as settings

class GameWorld:
    def __init__(self):
        #TODO: make it so 2 players can be initialized
        self.game_map = Map(file_location="map.txt")
        self.player = Player(30, 40, self.game_map)
        
    def update(self, keys):
        
        # Update the player
        self.player.update(keys, self.game_map)

        # When map will have more logic, it will be updated here
        # self.map.update()