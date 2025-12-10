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
        
        self.randFrames = random.randrange((2 * rl_settings.FRAME_SKIPPING_STEPS) * rl_settings.FRAME_SKIPPING_STEPS, (3 * rl_settings.FRAME_SKIPPING_STEPS) * rl_settings.FRAME_SKIPPING_STEPS)
        self.OptimizeRateSteps = 0
        self.SyncRateSteps = 0

        self.setup_agents()

        if rl_settings.LOAD_MODEL:
            self.load_agents("loads")
    
    def load_agents(self, file_path):
        for agentName in self.agentNames:
            policy_path = os.path.join(file_path, f"{agentName}_policy.pth")
            if os.path.exists(policy_path):
                print(policy_path)
                self.agents[agentName].policy_network.load_state_dict(torch.load(policy_path))
            else:
                print("policy network not loaded")
            
            if rl_settings.TRAINING_MODE:
                target_path = os.path.join(file_path, f"{agentName}_target.pth")
                if os.path.exists(target_path):
                    print(target_path)
                    self.agents[agentName].target_network.load_state_dict(torch.load(target_path))
                else:
                    print("target network not loaded")

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
            self.agents[agentName] = DQNAgent(stateSize=self.stateSize, action_size=rl_settings.ACTION_SPACE_SIZE, isTraining=self.isTraining)
            self.frameHistory[agentName] = deque(maxlen=rl_settings.FRAME_SKIPPING_STEPS * 2) # one for current and one for past state blocks
            self.episodeRewards[agentName] = {}
            self.lastAction[agentName] = 0

    def step_all_agents(self, statesForAgents, keys):
        currEpsilon = states.epsilon

        for agentName in self.agentNames:
            
            agentReward = statesForAgents[agentName][1]
            states.rewardsPerEpisode[agentName][states.episodeCount] += agentReward
            # if agentReward > 0:
            #     print(f"positive reward for agent: {agentReward}")
            #     print(f"total reward: {states.rewardsPerEpisode[agentName][states.episodeCount]}")

            # Only act once every n frames
            if states.episodeFrame % rl_settings.FRAME_SKIPPING_STEPS == 0:
                
                self.OptimizeRateSteps += 1
                self.SyncRateSteps += 1

                # Get the current state of the agent 
                currentState = statesForAgents[agentName][0]
                self.frameHistory[agentName].append(currentState)

                # If it's the first few frames of the episode, take random actions
                # untill the frame history fills up. 
                # This also makes the model learning offset to see different states
                if states.episodeFrame <= self.randFrames:
                    nextAgentAction = self.pick_random_action()
                    keys = self.action_to_input(agentName, nextAgentAction, keys) 
                    self.lastAction[agentName] = self.agents[agentName].float_to_device(self.action_to_idx(nextAgentAction))

                    continue

                # From here on we will need the recent frames and to start calculating the reward

                agentRewardTensor = self.agents[agentName].float_to_device(agentReward)
                # print(agentReward)

                # len(self.frameHistory[agentName]) has to be 2*rl_settings.FRAME_SKIPPING_STEPS
                recent_frames = list(self.frameHistory[agentName])

                lastStackedState = np.stack(recent_frames[:rl_settings.FRAME_SKIPPING_STEPS])
                currStackedState = np.stack(recent_frames[rl_settings.FRAME_SKIPPING_STEPS:])
                # print(len(recent_frames))
                # print(lastStackedState.shape)
                # print(currStackedState.shape)
                # print(f"First frame in history: {lastStackedState[0][:3]}...")
                # print(f"Last frame in history: {currStackedState[-1][:3]}...")


                # If not training, just get the action and move on
                if not rl_settings.TRAINING_MODE:
                    nextAgentAction = self.agents[agentName].step(currStackedState) 
                    self.lastAction[agentName] = self.agents[agentName].float_to_device(self.action_to_idx(nextAgentAction))
                    keys = self.action_to_input(agentName, nextAgentAction, keys) 
                    # print(f"next agent action: {nextAgentAction}")
                    continue

                experience = (lastStackedState, self.lastAction[agentName], currStackedState, agentRewardTensor, states.isTerminated)
                self.agents[agentName].memory.append(experience)

                # If the game was terminated, do nothing
                if states.isTerminated:
                    nextAgentAction = 0
                    keys = self.action_to_input(agentName, nextAgentAction, keys) 
                    self.lastAction[agentName] = self.agents[agentName].float_to_device(self.action_to_idx(nextAgentAction))

                    continue

                # If it's the first few episodes, collect dummy data to lessen the overfitting 
                # to the begginging of learning process
                if states.episodeCount < rl_settings.EXPERIENCE_COLLECTION_EPISODES:
                    nextAgentAction = self.pick_random_action()
                    keys = self.action_to_input(agentName, nextAgentAction, keys) 
                    self.lastAction[agentName] = self.agents[agentName].float_to_device(self.action_to_idx(nextAgentAction))

                    continue

                # After that, chose the action in an epsilon greedy fashion
                # TODO: Make the random choice seeded for replication
                if random.random() < currEpsilon:
                    nextAgentAction = self.pick_random_action()
                else:
                    nextAgentAction = self.agents[agentName].step(currStackedState) 

                self.lastAction[agentName] = self.agents[agentName].float_to_device(self.action_to_idx(nextAgentAction))
                keys = self.action_to_input(agentName, nextAgentAction, keys) 

            # All other frames repeat the last action
            else:
                keys = self.action_to_input(agentName, rl_settings.ACTIONS[self.lastAction[agentName]], keys) 
            
            # if agentReward > 0:
            #     print(f"eof agent step in episode {states.episodeCount}")
        # Every few steps update the models   
        if self.OptimizeRateSteps >= rl_settings.NETWORK_LEARN_RATE and rl_settings.TRAINING_MODE and states.episodeCount > rl_settings.EXPERIENCE_COLLECTION_EPISODES:
            self.OptimizeRateSteps = 0
            
            for agentName in self.agents:
                agent = self.agents[agentName]
                # If enough experience has been collected
                if len(agent.memory) > rl_settings.MINI_BATCH: 
                    mini_batch = agent.memory.sample(rl_settings.MINI_BATCH)
                    # Optimize the policy network
                    agent.optimize(mini_batch, agent.policy_network, agent.target_network)


        if states.isTerminated:
            self.post_episode_actions()
            # print("Terminated, starting episode count again")
        
        return keys
    
    def post_episode_actions(self):
        print(f"{states.episodeCount}: epsilon {states.epsilon}. episode reward {states.rewardsPerEpisode["player_one"][states.episodeCount]}")
        
        if states.episodeCount % 10000 == 0 and rl_settings.TRAINING_MODE:
            self.save_agents("saves", states.episodeCount)

        if global_settings.DEBUG_MODE:
            print(f"Episode: {states.episodeCount}")
            print(f"p1 episode reward: {states.rewardsPerEpisode["player_one"][states.episodeCount - 1]}")
            print(f"p2 episode reward: {states.rewardsPerEpisode["player_two"][states.episodeCount - 1]}")
            print(f"Running epsilon: {states.epsilon}")

        for agentName in self.agents.keys():
            agent = self.agents[agentName]
            if self.SyncRateSteps >= rl_settings.NETWORK_SYNC_RATE and len(agent.memory) > rl_settings.MINI_BATCH and rl_settings.TRAINING_MODE and states.episodeCount > rl_settings.EXPERIENCE_COLLECTION_EPISODES: 
                    self.total_steps = 0    
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

