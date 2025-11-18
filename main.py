import pygame
import sys
from Objects.GameWorld import GameWorld
from Objects.AgentController import AgentController
from Settings import rl_settings
import Objects.States as states


# Initialize Pygame
pygame.init()

# Game loop
clock = pygame.time.Clock()
running = True

# Initialize game world
gameWorld = GameWorld()

# Initialize the RL agents
# TODO: make this automatic based on settings
AgentController = AgentController(gameWorld.get_state_array_size(), ["player_one"])
states.rewardsPerEpisode["player_one"].append(rl_settings.START_REWARD)
states.rewardsPerEpisode["player_two"].append(rl_settings.START_REWARD)


while running:
    if states.endEpisode:
        # AgentController.post_episode_actions()
        states.endEpisode = False
    
    states.episodeFrame += 1
    if states.episodeFrame >= 600:
        states.isTerminated = True
    
    
    # Event handling
    k = pygame.key.get_pressed()
    
    # convert from the immutable base key array to a mutable array
    keys = {}

    # copy the keys from the base keys to the new key array
    # TODO: resolve the issue that arrow keys don't work, because their key code is not ASCII code 
    for key_code in range(len(k)):
        keys[key_code] = k[key_code]
        
    # Event loop
    for event in pygame.event.get():
        # Turn off logic
        if event.type == pygame.QUIT:
            running = False
    
    # Quick reset
    if keys[pygame.K_r]:
        states.isTerminated = True

    statesForAgents = {}

    if rl_settings.RL_CONTROL["player_one"]:
        # the 0 here looks awful, probably better to remake it with a string name
        statesForAgents["player_one"] = gameWorld.get_state_for_player(0) 
    if rl_settings.RL_CONTROL["player_two"]:
        # the 1 here looks awful, probably better to remake it with a string name
        statesForAgents["player_two"] = gameWorld.get_state_for_player(1) 

    # TODO: only get the agent key here
    keys = AgentController.step_all_agents(statesForAgents, keys)

    # Update everything in the game world
    gameWorld.update(keys)

    # TODO: update the agent memory here 

    if not rl_settings.TRAINING_MODE:
        # Draw the game world
        gameWorld.draw()
        
        clock.tick(60)  # 60 frames per second
    else:
        pass

# Clean up
pygame.quit()
sys.exit()
