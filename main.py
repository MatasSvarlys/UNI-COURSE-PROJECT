import pygame
import sys
import Camera
import Map
import Player
import global_settings as settings

# Initialize Pygame
pygame.init()

# Set up the screen
screen = pygame.display.set_mode((settings.SCREEN_WIDTH, settings.SCREEN_HEIGHT))

# Game loop
clock = pygame.time.Clock()
running = True

# Initialize player, camera, and map ONCE
player = Player.Player(10, settings.SCREEN_HEIGHT - settings.PLAYER_HEIGHT - 1.5*settings.TILE_SIZE_IN_SCREEN)
camera = Camera.Camera()
map = Map.Map(file_location="map.txt")


while running:
    # Event handling
    keys = pygame.key.get_pressed()
    
    
    # Camera mode toggle
    for event in pygame.event.get():
        # Turn off logic
        if event.type == pygame.QUIT:
            running = False
        

        # Debug camera mode toggle
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:  # Toggle camera mode with 'C'
                settings.CAMERA_MODE = not settings.CAMERA_MODE

    # Camera movement with WASD if camera mode is on
    if settings.CAMERA_MODE:
        if keys[pygame.K_d]:
            camera.move(10, 0)
        if keys[pygame.K_a]:
            camera.move(-10, 0)
        if keys[pygame.K_s]:
            camera.move(0, 10)
        if keys[pygame.K_w]:
            camera.move(0, -10)

    # Fill the background
    screen.fill(settings.BACKGROUND_COLOR)


    # Draw the map
    map.draw(screen, camera)
    
    
    # Draw the player

    player.update(keys)
    player.draw(screen, camera)
  
    
    # Update the display
    pygame.display.flip()
    clock.tick(60)  # 60 frames per second

# Clean up
pygame.quit()
sys.exit()
