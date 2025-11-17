from collections import deque
import json
import numpy as np
import pygame
import torch
from Settings import rl_settings
from Settings import global_settings
from Objects.DQNAgent import DQNAgent
import random
import os
import Objects.States as states

class AgentController:
    def __init__(self, stateSize, agentsNames):
        self.stateSize = stateSize

        self.agentNames = agentsNames
        self.agents = {}
        self.isTraining = rl_settings.TRAINING_MODE

        self.frameHistory = {}
        self.episodeRewards = {}
        self.lastAction = {}
        
        self.setup_agents()

        # self.load_agents("loads")
    
    def load_agents(self, file_path):
        for agentName in self.agentNames:
            policy_path = os.path.join(file_path, f"{agentName}_policy.pth")
            target_path = os.path.join(file_path, f"{agentName}_target.pth")
            if os.path.exists(policy_path) and os.path.exists(target_path):
                self.agents[agentName].policy_network.load_state_dict(torch.load(policy_path))
                self.agents[agentName].target_network.load_state_dict(torch.load(target_path))

    def save_agents(self, file_path, episode=None):
        os.makedirs(file_path, exist_ok=True)
        ep_suffix = f"_ep{int(episode)}" if episode is not None else ""
        for agentName, agent in self.agents.items():
            policy_path = os.path.join(file_path, f"{agentName}_policy{ep_suffix}.pth")
            target_path = os.path.join(file_path, f"{agentName}_target{ep_suffix}.pth")
            torch.save(agent.policy_network.state_dict(), policy_path)
            torch.save(agent.target_network.state_dict(), target_path)

    
    def setup_agents(self):
        for agentName in self.agentNames:
            if rl_settings.RL_CONTROL[agentName]:
                self.agents[agentName] = DQNAgent(stateSize=self.stateSize, action_size=rl_settings.ACTION_SPACE_SIZE, isTraining=self.isTraining)
                self.frameHistory[agentName] = deque(maxlen=rl_settings.FRAME_SKIPPING_STEPS * 2) # one for current and one for past state blocks
                self.episodeRewards[agentName] = {}
                self.lastAction[agentName] = 0

    def step_all_agents(self, statesForAgents, keys):
        currEpsilon = states.epsilon

        for agentName in self.agentNames:

            # Get the current state of the agent 
            currentState = statesForAgents[agentName][0]
            currentStateTensor = self.agents[agentName].float_list_to_device(currentState)
            self.frameHistory[agentName].append(currentState)


            # Since we get the reward every frame, might as well process it every frame too
            # This is not necessary and could be moved inside the if statement
            agentReward = statesForAgents[agentName][1]
            # print(agentReward)
            states.rewardsPerEpisode[agentName][states.episodeCount] += agentReward
            agentRewardTensor = self.agents[agentName].float_to_device(agentReward)


            
            # Only get a different action and store the experience after n steps (frames)
            if states.step % rl_settings.FRAME_SKIPPING_STEPS == 0:
                
                recent_frames = list(self.frameHistory[agentName])

                # Only process if we have enough frames
                if len(self.frameHistory[agentName]) == 2*rl_settings.FRAME_SKIPPING_STEPS:
                    # put the frames into a list
                    
                    # split the list in half
                    lastStackedState = np.concatenate(recent_frames[:rl_settings.FRAME_SKIPPING_STEPS])
                    currStackedState = np.concatenate(recent_frames[rl_settings.FRAME_SKIPPING_STEPS:])

                    # print(lastStackedState.shape, currStackedState.shape)
                    # SARS (State, Action, Reward, State) and if it is terminated
                    experience = (lastStackedState, self.lastAction[agentName], currStackedState, agentRewardTensor, states.isTerminated)
                    
                    # All of these are already in tensors for further processing
                    # State, action, new state, reward, terminated (make sure this is always accurate)
                    self.agents[agentName].memory.append(experience)
                
                # get the next action for the agent
                # TODO: Make the random choice seeded for replication
                if ((self.isTraining and random.random() < currEpsilon) or len(self.frameHistory[agentName]) < rl_settings.FRAME_SKIPPING_STEPS):
                    nextAgentAction = self.pick_random_action()
                else:
                    if len(self.frameHistory[agentName]) > rl_settings.FRAME_SKIPPING_STEPS:
                        currStackedState = np.concatenate(recent_frames[rl_settings.FRAME_SKIPPING_STEPS:])
                        nextAgentAction = self.agents[agentName].step(currStackedState) 
                    else:
                        nextAgentAction = self.pick_random_action()

                    # if states.isTerminated:
                    #    print(states.episodeCount, states.rewardsPerEpisode[agentName][states.episodeCount], currEpsilon)

                self.lastAction[agentName] = self.agents[agentName].float_to_device(self.action_to_idx(nextAgentAction))

                # change out the keys for that agent to the ones defined by an action
                keys = self.action_to_input(agentName, nextAgentAction, keys) 
            else:
                keys = self.action_to_input(agentName, rl_settings.ACTIONS[self.lastAction[agentName]], keys) 
        return keys
    
    def post_episode_actions(self):
        print(f"{states.episodeCount}: player one - {states.epsilon} - {states.rewardsPerEpisode["player_one"][states.episodeCount - 1]}")
        
        if states.episodeCount % 1000 == 0:
            self.save_agents("saves", states.episodeCount)

        if global_settings.DEBUG_MODE:
            print(f"Episode: {states.episodeCount}")
            print(f"p1 episode reward: {states.rewardsPerEpisode["player_one"][states.episodeCount - 1]}")
            print(f"p2 episode reward: {states.rewardsPerEpisode["player_two"][states.episodeCount - 1]}")
            print(f"Running epsilon: {states.epsilon}")

        for agentName in self.agents.keys():
            agent = self.agents[agentName]
            # If enough experience has been collected
            if len(agent.memory) > rl_settings.MINI_BATCH and rl_settings.TRAINING_MODE: 
                mini_batch = agent.memory.sample(rl_settings.MINI_BATCH)
                agent.optimize(mini_batch, agent.policy_network, agent.target_network)

                # TODO: figure this out
                if states.step > rl_settings.NETWORK_SYNC_RATE:
                    agent.target_network.load_state_dict(agent.policy_network.state_dict())


    def action_to_idx(self, action):
        actionSpace = rl_settings.ACTIONS
        try:
            # print(actionSpace.index(action))
            return actionSpace.index(action)
        except ValueError:
            return 0
        
    def pick_random_action(self):
        actionSpace = rl_settings.ACTIONS
        action = actionSpace[random.randint(0, 5)]
        return action

    def action_to_input(self, player_name, action, keys):
        config_file = "./PlayerKeybinds/p1.json" if player_name == "player_one" else "./PlayerKeybinds/p2.json"

        # Load key mappings from JSON file
        with open(config_file, 'r') as f:
            key_mappings = json.load(f)
        
        # Block all keys defined in the config file
        for _, key_name in key_mappings.items():
            key_code = getattr(pygame, key_name)
            keys[key_code] = False

        match action:
            case "NOOP":
                pass
            case "LEFT":
                keys[getattr(pygame, key_mappings["MOVE_LEFT"])] = True
            case "RIGHT":
                keys[getattr(pygame, key_mappings["MOVE_RIGHT"])] = True
            case "JUMP":
                keys[getattr(pygame, key_mappings["JUMP"])] = True
            case "LEFT_JUMP":
                keys[getattr(pygame, key_mappings["MOVE_LEFT"])] = True
                keys[getattr(pygame, key_mappings["JUMP"])] = True
            case "RIGHT_JUMP":
                keys[getattr(pygame, key_mappings["MOVE_RIGHT"])] = True
                keys[getattr(pygame, key_mappings["JUMP"])] = True

        return keys

