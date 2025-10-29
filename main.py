import pygame
import sys
from Objects.GameWorld import GameWorld
from Objects.AgentController import AgentController
from Settings import rl_settings

# Flags
flags = [
    False # terminated
]

# Initialize Pygame
pygame.init()

# Game loop
clock = pygame.time.Clock()
running = True

# Initialize game world
gameWorld = GameWorld()

# Initialize the RL agents
AgentController = AgentController(gameWorld.get_state_array_size())

while running:
    # Event handling
    k = pygame.key.get_pressed()
    
    # convert from the immutable base key array to a mutable array
    keys = {}

    # copy the keys from the base keys to the new key array
    for key_code in range(len(k)):
        keys[key_code] = k[key_code]
        
    # Event loop
    for event in pygame.event.get():
        # Turn off logic
        if event.type == pygame.QUIT:
            running = False
    
    # Quick reset
    if keys[pygame.K_r]:
        flags[0] = True

    statesForAgents = {}

    if rl_settings.RL_CONTROL["player_one"]:
        # the 0 here looks awful, probably better to remake it with a string name
        statesForAgents["player_one"] = gameWorld.get_state_for_player(0) 
    if rl_settings.RL_CONTROL["player_two"]:
        # the 0 here looks awful, probably better to remake it with a string name
        statesForAgents["player_two"] = gameWorld.get_state_for_player(1) 

    keys = AgentController.step_all_agents(statesForAgents, keys)

    # Update everything in the game world
    gameWorld.update(keys, flags)

    # Draw the game world
    gameWorld.draw()

    clock.tick(60)  # 60 frames per second

# Clean up
pygame.quit()
sys.exit()
