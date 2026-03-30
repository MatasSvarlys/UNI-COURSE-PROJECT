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
AgentController = AgentController()

while running:
    
    states.episodeFrame += 1
    if states.episodeFrame >= 1200:
        states.isTerminated = True
    
    if states.episodeCount >= 1000000:
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
            if rl_settings.CLASSIC_MODE:
                statesForAgents[name] = gameWorld.get_state_screenshot()
            else:
                statesForAgents[name] = gameWorld.get_player_observation(idx)
    
    # 2. Get predicted action 
    agentActions = AgentController.step_all_agents(statesForAgents)

    for idx, name in enumerate(rl_settings.RL_CONTROL.keys()):
        if rl_settings.RL_CONTROL[name]:
            playerActions[idx] = agentActions[name]
        else:
            playerActions[idx] = keys_to_action(idx, k)

    # 3. Take the action and calculate reward
    gameWorld.update(playerActions)

    for idx, name in enumerate(rl_settings.RL_CONTROL.keys()):
        if rl_settings.RL_CONTROL[name]:
            agentReward = gameWorld.get_reward(idx)
            states.episodeReward[name] += agentReward

    # 4. Store the experience on the frames where the action was actually taken
    if (states.episodeFrame + AgentController.randFrames) % rl_settings.FRAMES_PER_STEP == 0:
        for idx, name in enumerate(rl_settings.RL_CONTROL.keys()):
            if rl_settings.RL_CONTROL[name]:
                if AgentController.lastStackedState[name] is not None:
                    AgentController.save_experience(name, AgentController.lastStackedState[name], AgentController.lastAction[name], AgentController.stackedState[name], gameWorld.get_reward(idx), states.isTerminated)
                if len(AgentController.frameHistory[name]) == rl_settings.FRAMES_PER_STEP:
                    AgentController.lastStackedState[name] = AgentController.stackedState[name]

    if states.isTerminated:
        AgentController.post_episode_actions()
        gameWorld.reset()
        states.startNewEpisode()


    states.epsilon = max(states.epsilon - rl_settings.EPSILON_DECAY, rl_settings.MIN_EPSILON)

    if not global_settings.HEADLESS_MODE:
        gameWorld.draw()
        clock.tick(60)  # 60 frames per second
        

# Clean up
pygame.quit()
sys.exit()
