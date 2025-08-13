import pygame
from Settings import global_settings as settings
from Settings import tile_settings
from Settings import map_settings

class Map:
    def __init__(self, file_location=None):

        # If a file location is provided, load the map from the file
        if file_location:
            with open(file_location, 'r') as f:
                
                # ------------- [depricated] -------------
                # Read the first line to get width and height 
                # first_line = f.readline().strip()

                # Parse width and height from the first line
                # self.width, self.height = map(int, first_line.split())
                # ----------------------------------------


                # For testing reasons, I'll set a constant map size in the map settings
                self.width = map_settings.MAP_WIDTH
                self.height = map_settings.MAP_HEIGHT

                
                # Init the map data
                self.map_data = []

                # Read the the rest to fill the map data
                for _ in range(self.height):
                    # Read a row and split into tiles. This is stored as an array of integers
                    # The delimiter is a space, might want to change it later
                    row = f.readline().strip().split()
                    
                    # Take the int value of each tile and append to the map data. Map data is a 2D array
                    self.map_data.append([int(tile) for tile in row])

        # If no file location is provided, initialize an empty map
        else:
            self.width = 0
            self.height = 0
            self.map_data = []


    def set_tile(self, x, y, value):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.map_data[y][x] = value

    def get_tile(self, x, y):
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.map_data[y][x]
        return None

    def log(self):
        for row in self.map_data:
            print(" ".join(str(tile) for tile in row))

    def get_grid(self):
        return self.map_data
    
    # Draw map with camera offset
    def draw(self, screen, camera):

        camera_x, camera_y = camera.get_position()

        for y, row in enumerate(self.get_grid()):
            for x, tile in enumerate(row):

                # To match the way we itterate through the map, 
                # we need to calculate the screen position of the tile
                # starting from the top left corner of the screen
                                
                screen_x = x * tile_settings.TILE_SIZE - camera_x
                screen_y = (y+1) * tile_settings.TILE_SIZE - camera_y
                

                # maybe it's better to draw 0 tiles too idk yet
                if tile != 0:
                    tile_rect = pygame.Rect(screen_x, screen_y - tile_settings.TILE_SIZE, tile_settings.TILE_SIZE, tile_settings.TILE_SIZE)
                    color = tile_settings.TILE_TYPE_MAP.get(tile, {"color": (0, 0, 0)})["color"]
                    pygame.draw.rect(screen, color, tile_rect)