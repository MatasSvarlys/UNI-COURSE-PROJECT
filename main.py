import pygame
import sys
from Objects.Camera import Camera
from Objects.GameWorld import GameWorld
from Settings import global_settings as settings


# Initialize Pygame
pygame.init()

# Set up the screen
window = pygame.display.set_mode((settings.WINDOW_WIDTH, settings.WINDOW_HEIGHT))

# Game loop
clock = pygame.time.Clock()
running = True

# Initialize game world and the main camera to display everything
camera = Camera((settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT))
game_world = GameWorld()

while running:
    # Event handling
    keys = pygame.key.get_pressed()
    
    
    # Event loop
    for event in pygame.event.get():
        # Turn off logic
        if event.type == pygame.QUIT:
            running = False
    

    # Fill the background
    window.fill(settings.BACKGROUND_COLOR)

    # Update everything in the game world
    game_world.update(keys)

    # Set the camera position
    # TODO: This is broken
    
    # Draw the game world
    camera.follow_with_offset(game_world.player.hitbox, offset_x=0, offset_y=-settings.SCREEN_HEIGHT // 4)
    camera.draw_world(game_world, window)

    # camera.render_to_window(window, game_world.player.hitbox.x // 2, game_world.player.hitbox.y // 2)

    # Update the display
    pygame.display.flip()
    clock.tick(60)  # 60 frames per second

# Clean up
pygame.quit()
sys.exit()
