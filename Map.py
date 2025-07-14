import pygame
import global_settings as settings
import tile_settings

class Map:
    def __init__(self, file_location=None):
        # If a file location is provided, load the map from the file
        if file_location:
            with open(file_location, 'r') as f:
                # Read the first line to get width and height
                first_line = f.readline().strip()

                # Parse width and height from the first line
                self.width, self.height = map(int, first_line.split())
                
                # Init the map data
                self.map_data = []

                # Read the the rest to fill the map data
                for _ in range(self.height):
                    # Read a row and split into tiles. This is stored as an array of integers
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
    
    def draw(self, screen, camera):
        # Draw map with camera offset
        camera_x, camera_y = camera.get_position()
        for y, row in enumerate(self.get_grid()):
            for x, tile in enumerate(row):
                screen_x = x * settings.TILE_SIZE_IN_SCREEN - camera_x
                screen_y = settings.SCREEN_HEIGHT - ((y + 1) * settings.TILE_SIZE_IN_SCREEN) - camera_y
                
                # Only render visible tiles
                if (-settings.TILE_SIZE_IN_SCREEN <= screen_x <= settings.SCREEN_WIDTH and 
                    -settings.TILE_SIZE_IN_SCREEN <= screen_y <= settings.SCREEN_HEIGHT):
                    rect = pygame.Rect(screen_x, screen_y, settings.TILE_SIZE_IN_SCREEN, settings.TILE_SIZE_IN_SCREEN)
                    pygame.draw.rect(screen, tile_settings.TILE_TYPES[tile]["color"], rect)