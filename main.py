import pygame
import sys
import Player
import global_settings as settings

# Initialize Pygame
pygame.init()

# Set up the screen
screen = pygame.display.set_mode((settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT))

# Game loop
clock = pygame.time.Clock()
running = True

player = Player.Player(100, settings.SCREEN_HEIGHT - settings.PLAYER_HEIGHT - 50)


while running:
    # Turn off logic
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Fill the screen with a color
    screen.fill(settings.BACKGROUND_COLOR)

    keys = pygame.key.get_pressed()
    player.update(keys)
    player.draw(screen)

    # Update the display
    pygame.display.flip()
    clock.tick(60)  # 60 frames per second

# Clean up
pygame.quit()
sys.exit()
