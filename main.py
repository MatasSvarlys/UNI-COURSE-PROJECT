import numpy as np
import pygame
import sys
from Objects.GameWorld import GameWorld
from Objects.AgentController import AgentController
from Settings import rl_settings
from Settings import global_settings
import Objects.States as states
from helper_functions.keyToAction import keys_to_action

# Initialize Pygame
pygame.init()

# Game loop
clock = pygame.time.Clock()
running = True

# Initialize game world``
gameWorld = GameWorld()

# Initialize the RL agents
# TODO: make this automatic based on settings
AgentController = AgentController(gameWorld.get_state_array_size())

while running:
    
    states.episodeFrame += 1
    if states.episodeFrame >= 600:
        states.isTerminated = True
    
    if states.episodeCount >= 100:
        pygame.quit()
        sys.exit()
    
    # Event handling
    k = pygame.key.get_pressed()
    

    # Event loop
    for event in pygame.event.get():
        # Turn off logic
        if event.type == pygame.QUIT:
            running = False
    
    # Quick reset
    if k[pygame.K_r] and not states.rKeyPressed:
        states.isTerminated = True
        states.rKeyPressed = True
    elif not k[pygame.K_r]:
        states.rKeyPressed = False


    statesForAgents = {}
    playerActions = {}

    # 1. Get state for each agent
    for idx, name in enumerate(rl_settings.RL_CONTROL.keys()):
        if rl_settings.RL_CONTROL[name]:
            statesForAgents[name] = gameWorld.get_state_for_player(idx) 
    
    # 2. Get predicted action 
    agentActions = AgentController.step_all_agents(statesForAgents)

    for idx, name in enumerate(rl_settings.RL_CONTROL.keys()):
        if rl_settings.RL_CONTROL[name]:
            playerActions[idx] = agentActions[name]
        else:
            playerActions[idx] = keys_to_action(idx, k)

    # 3. Take the action and calculate reward
    gameWorld.update(playerActions)


    # 4. Store the experience
    for idx, name in enumerate(rl_settings.RL_CONTROL.keys()):
        if rl_settings.RL_CONTROL[name]:
            
            agentReward = gameWorld.get_reward(idx)
            states.episodeReward[name] += agentReward

            AgentController.frameHistory[name].append(statesForAgents[name][0])
            # TODO: maybe only do the calculations when needed instead of leaving the check in save_experience
            if len(AgentController.frameHistory[name]) >= rl_settings.STEPS_PER_ACTION * 2:
                recent_frames = list(AgentController.frameHistory[name])
                lastStackedState = np.stack(recent_frames[:rl_settings.STEPS_PER_ACTION])
                currStackedState = np.stack(recent_frames[rl_settings.STEPS_PER_ACTION:])
                AgentController.save_experience(name, lastStackedState, playerActions[idx], currStackedState, statesForAgents[name][1], states.isTerminated)

    if states.isTerminated:
        AgentController.post_episode_actions()
        gameWorld.reset()
        states.startNewEpisode()


    if not global_settings.HEADLESS_MODE:
        gameWorld.draw()
        clock.tick(60)  # 60 frames per second
        

# Clean up
pygame.quit()
sys.exit()
