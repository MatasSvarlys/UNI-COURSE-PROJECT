import random

import numpy as np
import pygame
import sys
import time
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

cumulative_reward = {name: 0 for name in rl_settings.RL_CONTROL.keys()}

# init for the states with the first state
statesForAgents = {}

for idx, name in enumerate(rl_settings.RL_CONTROL.keys()):
    if rl_settings.RL_CONTROL[name]:
        if rl_settings.CLASSIC_MODE:
            statesForAgents[name] = gameWorld.get_state_screenshot()
        else:
            statesForAgents[name] = gameWorld.get_player_observation(idx)

# random offset so the model doesnt always see the same frame
randFrames = random.randrange(1, rl_settings.FRAMES_PER_STEP)

start_training_time = time.time()

while running:

    if gameWorld.captureOccured:
        states.framesLeft = rl_settings.MAX_FRAMES
        gameWorld.captureOccured = False

    states.framesLeft -= 1
    states.episodeFrame += 1
    if states.framesLeft <= 0:
        states.isTerminated = True
    
    if states.episodeCount >= rl_settings.MAX_EPISODES:
        AgentController.shutdown_logging()
        running = False
    
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

    playerActions = {}
    agentActions = {}

    # For the first frame in a new map, fill the history with the same frame to avoid 
    # leaking from last map and passing a non-full history
    for _, agentName in enumerate(rl_settings.RL_CONTROL.keys()):
        if rl_settings.RL_CONTROL[agentName]:
            if len(AgentController.frameHistory[agentName]) == 0:
                for _ in range(rl_settings.STEPS_PER_ACTION):
                    AgentController.frameHistory[agentName].append(statesForAgents[agentName])



    # 1. If its time to make an action, choose a new one 
    if (states.episodeFrame + randFrames) % rl_settings.FRAMES_PER_STEP == 0:
        agentActions = AgentController.step_all_agents()
    else:
        # otherwise, use the last one
        for _, agentName in enumerate(rl_settings.RL_CONTROL.keys()):
            if rl_settings.RL_CONTROL[agentName]:
                agentActions[agentName] = AgentController.lastAction[agentName]
        
    # 1.2 convert the action into input
    for idx, name in enumerate(rl_settings.RL_CONTROL.keys()):
        if rl_settings.RL_CONTROL[name]:
            playerActions[idx] = agentActions[name]
        else:
            playerActions[idx] = keys_to_action(idx, k)

    # 2. update the world
    gameWorld.update(playerActions)
    
    # 3. Get state for each agent after the map has changed
    for idx, name in enumerate(rl_settings.RL_CONTROL.keys()):
        if rl_settings.RL_CONTROL[name]:
            if rl_settings.CLASSIC_MODE:
                statesForAgents[name] = gameWorld.get_state_screenshot()
            else:
                statesForAgents[name] = gameWorld.get_player_observation(idx)
    
            # 3.2 append the current frame to the history
            AgentController.frameHistory[name].append(statesForAgents[name])

            # 3.3 calculate the reward from the update
            agentReward = gameWorld.get_reward(idx)
            cumulative_reward[name] += agentReward
            states.episodeReward[name] += agentReward


    # 4. if the new state is terminal or enough frames have passed 
    if gameWorld.captureOccured or states.isTerminated or (states.episodeFrame + randFrames) % rl_settings.FRAMES_PER_STEP == 0:
        for idx, name in enumerate(rl_settings.RL_CONTROL.keys()):
            if rl_settings.RL_CONTROL[name] and AgentController.agents[name].learning_enabled:
                # append the transition to the N step buffer
                currentState = list(AgentController.frameHistory[name])
                transition = (currentState, agentActions[name], cumulative_reward[name])
                AgentController.nStepBuffers[name].append(transition)
                
                # if it was terminal, save all the intermediate steps that lead to termination
                if states.isTerminated or gameWorld.captureOccured:
                    while len(AgentController.nStepBuffers[name]) > 0:
                        # only save with the terminal flag as true when the player actually catches the other player
                        AgentController.save_from_nstep(name, currentState, gameWorld.captureOccured)
                        AgentController.nStepBuffers[name].popleft()

                # otherwise just save the full nstep transition
                elif len(AgentController.nStepBuffers[name]) == rl_settings.N_STEP_LENGTH:
                    AgentController.save_from_nstep(name, currentState, False)
                
            cumulative_reward[name] = 0


    
    if gameWorld.captureOccured:
        gameWorld.reset()
        gameWorld.captureOccured = False
        states.framesLeft += 50

    if states.isTerminated:
        AgentController.post_episode_actions()
        gameWorld.reset()
        states.startNewEpisode()
        randFrames = random.randrange(1, rl_settings.FRAMES_PER_STEP)
        
        if rl_settings.RL_CONTROL["player_one"] and rl_settings.RL_CONTROL["player_two"]:
            for idx, name in enumerate(rl_settings.RL_CONTROL.keys()):
                agent = AgentController.agents[name]
                
                if states.episodeCount < rl_settings.SINGLE_AGENT_TRAINING_EPISODES:
                    # Catcher (idx 0) learns, Runner (idx 1) frozen
                    agent.learning_enabled = (idx == 0)
                elif states.episodeCount < rl_settings.SINGLE_AGENT_TRAINING_EPISODES * 2:
                    # Runner learns, Catcher frozen
                    agent.learning_enabled = (idx == 1)
                else:
                    # Both learn
                    agent.learning_enabled = True


    if not global_settings.HEADLESS_MODE:
        gameWorld.draw()
        clock.tick(60)  # 60 frames per second
        


end_training_time = time.time()
total_seconds = int(end_training_time - start_training_time)

# Format the time
hours, remainder = divmod(total_seconds, 3600)
minutes, seconds = divmod(remainder, 60)
time_str = f"{hours:02}:{minutes:02}:{seconds:02}"

print(f"\n" + "="*30)
print(f"TRAINING COMPLETE")
print(f"Total Episodes: {states.episodeCount}")
print(f"Total Time: {time_str}")
print("="*30)

# Clean up
pygame.quit()
sys.exit()
