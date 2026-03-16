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
        # TODO: use the global var instead of passing map data
        self.blockGrid = map_data_raw
        self.grid_width = len(map_data_raw[0]) if map_data_raw else 0
        self.grid_height = len(map_data_raw)

        self.p1StartPos = self.getPlayerPosition(map_data_raw, 2)
        self.p2StartPos = self.getPlayerPosition(map_data_raw, 3)

        self.collision_rects = self.calculate_collision_rects(map_data_raw)
        self.drawRects = self.calculate_draw_rects(map_data_raw)

    def getPlayerPosition(self, map_data_raw, playerNumber):
        for y_coord, row in enumerate(map_data_raw):
            for x_coord, tile in enumerate(row):
                if tile == playerNumber:
                    playerX = x_coord * map_settings.TILE_SIZE
                    playerY = y_coord * map_settings.TILE_SIZE   

        return (playerX, playerY)
    

    def calculate_draw_rects(self, map_data_raw):
        drawRects = []
        tileTypeMap = map_settings.TILE_TYPE_MAP
        for y_coord, row in enumerate(map_data_raw):
            for x_coord, tile in enumerate(row):
                if tile == 2 or tile == 3:
                    tile = 0
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

    def world_to_grid_coordinates(self, world_x, world_y):
        # convert world coordinates to map grid coordinates
        grid_x = int(world_x // map_settings.TILE_SIZE)
        grid_y = int(world_y // map_settings.TILE_SIZE)
        return grid_x, grid_y
    
    def get_nearby_collision_rects(self, rect, search_radius=2):
        nearby_rects = []
        
        # Get aprox grid position of the player
        player_grid_x, player_grid_y = self.world_to_grid_coordinates(rect.centerx, rect.centery)
        
        # Search surrounding grid cells
        for dy in range(-search_radius, search_radius + 1):
            for dx in range(-search_radius, search_radius + 1):
                grid_x = player_grid_x + dx
                grid_y = player_grid_y + dy
                
                # Check if grid coordinates are valid
                if (0 <= grid_x < self.grid_width and 
                    0 <= grid_y < self.grid_height):
                    
                    tile = self.blockGrid[grid_y][grid_x]
                    
                    # Check if this tile is solid and should have a collision rect
                    if (tile in map_settings.TILE_TYPE_MAP and 
                        map_settings.TILE_TYPE_MAP[tile]["solid"]):
                        
                        # Calculate the world position
                        world_x = grid_x * map_settings.TILE_SIZE
                        world_y = grid_y * map_settings.TILE_SIZE
                        rect = pygame.Rect(world_x, world_y, map_settings.TILE_SIZE, map_settings.TILE_SIZE)
                        nearby_rects.append(rect)
        
        return nearby_rects
    
    def draw_to_surface(self, surface):
        for tile in self.drawRects:
            pygame.draw.rect(
                surface, # what surface
                tile[1], # color
                tile[0] # rect
                )
            
        return surface