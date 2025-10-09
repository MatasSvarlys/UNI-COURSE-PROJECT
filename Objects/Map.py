import pygame
from Settings import map_settings
from typing import List

class Map:
    collision_rects: List[pygame.Rect]  # Just the solid rectangles
    drawRects: List[tuple[pygame.Rect, tuple[int, int, int]]] # All map rectagnles and their color (TODO: add sprites instead of colors)

    def __init__(self, file_location = None):
        map_data_raw = []
        
        # If a file location is provided, load the map from the file
        if file_location:
            with open(file_location, 'r') as f:
                for line in f:
                    row = line.strip().split()
                    map_data_raw.append([int(tile) for tile in row])
        
        # TODO: recalculate these in an update funciton if blocks move
        self.collision_rects = self.calculate_collision_rects(map_data_raw)
        self.drawRects = self.calculate_draw_rects(map_data_raw)

    def calculate_draw_rects(self, map_data_raw):
        drawRects = []
        tileTypeMap = map_settings.TILE_TYPE_MAP
        for y_coord, row in enumerate(map_data_raw):
            for x_coord, tile in enumerate(row):
                if tile in tileTypeMap:
                    # Calculate the world position of the tile
                    world_x = x_coord * map_settings.TILE_SIZE
                    world_y = y_coord * map_settings.TILE_SIZE
                    
                    # Create a rect for the tile
                    rect = pygame.Rect(world_x, world_y, map_settings.TILE_SIZE, map_settings.TILE_SIZE)
                    color = tileTypeMap[tile]["color"]
                    drawRects.append((rect, color))
        
        return drawRects


    def calculate_collision_rects(self, raw_data):
        collisionRects = []
        tileTypeMap = map_settings.TILE_TYPE_MAP

        for y_coord, row in enumerate(raw_data):
            for x_coord, tile in enumerate(row):
                if tile in tileTypeMap and tileTypeMap[tile]["solid"]:
                    # Calculate the world position of the tile
                    world_x = x_coord * map_settings.TILE_SIZE
                    world_y = (y_coord + 1) * map_settings.TILE_SIZE - map_settings.TILE_SIZE
                    
                    # Create a rect for the tile
                    rect = pygame.Rect(world_x, world_y, map_settings.TILE_SIZE, map_settings.TILE_SIZE)
                    collisionRects.append(rect)
        
        return collisionRects

    def draw_to_surface(self, surface):
        for tile in self.drawRects:
            pygame.draw.rect(
                surface, # what surface
                tile[1], # color
                tile[0] # rect
                )
            
        return surface