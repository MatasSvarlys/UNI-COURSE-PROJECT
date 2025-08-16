import pygame
from Settings import map_settings
from typing import List

class Map:
    map_data_raw: List[List[int]] # int is the tile ID
    collision_rects: List[pygame.Rect]  # Just the solid rectangles
    draw_rects: List[tuple[pygame.Rect, tuple[int, int, int]]] # All map rectagnles and their color (TODO: add sprites instead of colors)
    tile_size: int
    tile_type_map: dict

    def __init__(self, file_location = None):
        self.map_data_raw = []
        self.collision_rects = []

        # Why not use the tile settings?
        # So I don't import more settings than I need to
        self.tile_size = map_settings.TILE_SIZE
        self.tile_type_map = map_settings.TILE_TYPE_MAP
        
        # If a file location is provided, load the map from the file
        if file_location:
            with open(file_location, 'r') as f:
                for line in f:
                    row = line.strip().split()
                    self.map_data_raw.append([int(tile) for tile in row])
        
        self.calculate_collision_rects(self.map_data_raw)

        self.calculate_draw_rects(self.map_data_raw)




    def calculate_draw_rects(self, map_data_raw):
        self.draw_rects = []
        for y_coord, row in enumerate(map_data_raw):
            for x_coord, tile in enumerate(row):
                if tile in self.tile_type_map:
                    # Calculate the world position of the tile
                    world_x = x_coord * self.tile_size
                    world_y = (y_coord + 1) * self.tile_size - self.tile_size
                    
                    # Create a rect for the tile
                    rect = pygame.Rect(world_x, world_y, self.tile_size, self.tile_size)
                    color = self.tile_type_map[tile]["color"]
                    self.draw_rects.append((rect, color))


    def calculate_collision_rects(self, raw_data):

        for y_coord, row in enumerate(raw_data):
            for x_coord, tile in enumerate(row):
                if tile in self.tile_type_map and self.tile_type_map[tile]["solid"]:
                    # Calculate the world position of the tile
                    world_x = x_coord * self.tile_size
                    world_y = (y_coord + 1) * self.tile_size - self.tile_size
                    
                    # Create a rect for the tile
                    rect = pygame.Rect(world_x, world_y, self.tile_size, self.tile_size)
                    self.collision_rects.append(rect)
