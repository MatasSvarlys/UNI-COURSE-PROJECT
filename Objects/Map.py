import pygame
from Settings import global_settings as settings
from Settings import tile_settings
from Settings import map_settings

class Map:
    def __init__(self, file_location=None):
        self.map_data = []

        # If a file location is provided, load the map from the file
        if file_location:
            with open(file_location, 'r') as f:
                for line in f:
                    row = line.strip().split()
                    self.map_data.append([int(tile) for tile in row])

    # Draw map with camera offset
    def draw(self, window, camera):

        camera_x, camera_y = camera.get_position()

        for y_coord, row in enumerate(self.map_data):
            for x_coord, tile in enumerate(row):

                # Now that we have each tiles coordinates, we can calculate the screen position
                screen_x = x_coord * tile_settings.TILE_SIZE - camera_x
                screen_y = (y_coord+1) * tile_settings.TILE_SIZE - camera_y
                

                # maybe it's better to draw 0 tiles too idk yet
                if tile != 0:
                    tile_rect = pygame.Rect(screen_x, screen_y - tile_settings.TILE_SIZE, tile_settings.TILE_SIZE, tile_settings.TILE_SIZE)
                    color = tile_settings.TILE_TYPE_MAP.get(tile, {"color": (0, 0, 0)})["color"]
                    pygame.draw.rect(window, color, tile_rect)