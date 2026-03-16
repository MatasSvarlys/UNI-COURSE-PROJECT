import os
import pygame
from Settings import map_settings

# Initialize Pygame
pygame.init()

def get_player_position(map_data_raw, player_number):
    """Find player starting position in map data"""
    for y_coord, row in enumerate(map_data_raw):
        for x_coord, tile in enumerate(row):
            if tile == player_number:
                player_x = x_coord * map_settings.TILE_SIZE
                player_y = y_coord * map_settings.TILE_SIZE
                return (player_x, player_y)
    return None

def load_map_from_file(file_path):
    """Load map data from file"""
    map_data_raw = []
    with open(file_path, 'r') as f:
        for line in f:
            row = line.strip().split()
            map_data_raw.append([int(tile) for tile in row])
    return map_data_raw

def draw_map(map_data_raw):
    """Draw the map and return the surface"""
    height = len(map_data_raw)
    width = len(map_data_raw[0]) if map_data_raw else 0
    
    # Create surface for the map
    surface = pygame.Surface((width * map_settings.TILE_SIZE, 
                             height * map_settings.TILE_SIZE))
    surface.fill((0, 0, 0))  # Black background
    
    tile_type_map = map_settings.TILE_TYPE_MAP
    
    # Draw tiles
    for y_coord, row in enumerate(map_data_raw):
        for x_coord, tile in enumerate(row):
            # Replace player markers with empty tiles for visualization
            display_tile = tile
            if tile == 2 or tile == 3:
                display_tile = 0
            
            if display_tile in tile_type_map:
                world_x = x_coord * map_settings.TILE_SIZE
                world_y = y_coord * map_settings.TILE_SIZE
                color = tile_type_map[display_tile]["color"]
                
                rect = pygame.Rect(world_x, world_y, 
                                  map_settings.TILE_SIZE, 
                                  map_settings.TILE_SIZE)
                pygame.draw.rect(surface, color, rect)
    
    return surface

def draw_player_markers(surface, p1_pos, p2_pos):
    """Draw player starting positions on the surface"""
    if p1_pos:
        # Player 1 - Green square
        rect = pygame.Rect(p1_pos[0], p1_pos[1], 
                          map_settings.TILE_SIZE, 
                          map_settings.TILE_SIZE)
        pygame.draw.rect(surface, (0, 255, 0), rect)
    
    if p2_pos:
        # Player 2 - Red square
        rect = pygame.Rect(p2_pos[0], p2_pos[1], 
                          map_settings.TILE_SIZE, 
                          map_settings.TILE_SIZE)
        pygame.draw.rect(surface, (255, 0, 0), rect)

def save_map_image(map_file):
    """Load a map and save it as an image with player positions marked"""
    # Load map data
    map_data = load_map_from_file(map_file)
    
    # Get player positions
    p1_pos = get_player_position(map_data, 2)
    p2_pos = get_player_position(map_data, 3)
    
    # Draw map
    surface = draw_map(map_data)
    
    # Draw player markers
    draw_player_markers(surface, p1_pos, p2_pos)
    
    # Create output filename
    map_name = os.path.splitext(os.path.basename(map_file))[0]
    output_dir = "map_images"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{map_name}_start_positions.png")
    
    # Save image
    pygame.image.save(surface, output_path)
    print(f"Saved: {output_path}")
    
    return output_path

def main():
    """Main function to process all maps"""
    maps_dir = "maps"
    
    # Check if maps directory exists
    if not os.path.exists(maps_dir):
        print(f"Error: '{maps_dir}' directory not found!")
        return
    
    # Get all map files
    map_files = []
    for file in os.listdir(maps_dir):
        if file.endswith(".txt") and file.startswith("map"):
            map_files.append(os.path.join(maps_dir, file))
    
    if not map_files:
        print("No map files found!")
        return
    
    print(f"Found {len(map_files)} map files")
    print("Processing maps...")
    
    # Process each map
    for map_file in sorted(map_files):
        save_map_image(map_file)
    
    print(f"\nAll done! Images saved to 'map_images' directory")
    print(f"Legend: Green square = Player 1 start, Red square = Player 2 start")

if __name__ == "__main__":
    main()
    pygame.quit()