import pygame
import sys
from Objects.GameWorld import GameWorld


# Initialize Pygame
pygame.init()

# Game loop
clock = pygame.time.Clock()
running = True

# Initialize game world
game_world = GameWorld()

while running:
    # Event handling
    keys = pygame.key.get_pressed()
    
    
    # Event loop
    for event in pygame.event.get():
        # Turn off logic
        if event.type == pygame.QUIT:
            running = False
    

    # Update everything in the game world
    game_world.update(keys)

    # Draw the game world
    game_world.draw()
    
    # Update the display
    pygame.display.flip()
    clock.tick(60)  # 60 frames per second

# Clean up
pygame.quit()
sys.exit()
