import json
import pygame
import torch
from Settings import rl_settings
from Settings import global_settings
from Objects.DQNAgent import DQNAgent
import random
import Objects.States as states

class AgentController:
    def __init__(self, stateSize):
        self.stateSize = stateSize
        self.agents = {}
        self.isTraining = True
        
        self.setup_agents()

        self.episodeRewards = {}
        for agent in self.agents.keys():
            self.episodeRewards[agent] = {}

        self.lastAction = {}
        for agent in self.agents.keys():
            self.lastAction[agent] = 0


    def setup_agents(self):
        if rl_settings.RL_CONTROL["player_one"]:
            self.agents["player_one"] = DQNAgent(stateSize=self.stateSize, action_size=rl_settings.ACTION_SPACE_SIZE, isTraining=self.isTraining)
        if rl_settings.RL_CONTROL["player_two"]:
            self.agents["player_two"] = DQNAgent(stateSize=self.stateSize, action_size=rl_settings.ACTION_SPACE_SIZE, isTraining=self.isTraining)
    
    def step_all_agents(self, statesForAgents, keys):
        currEpsilon = states.epsilon

        for agent in self.agents.keys():
            # Only get a different action after n steps (frames)
            if states.step % rl_settings.FRAME_SKIPPING_STEPS == 0:
                prevState = self.agents[agent].prevState
                
                # get the next action for the agent
                if (self.isTraining and random.random() < currEpsilon):
                    nextAgentAction = self.pick_random_action()
                    self.agents[agent].prevState = self.agents[agent].state_to_device(statesForAgents[agent][0])
                else:
                    nextAgentAction = self.agents[agent].step(statesForAgents[agent][0]) 
                
                agentReward = statesForAgents[agent][1]
                agentRewardTensor = self.agents[agent].action_to_device(agentReward)

                states.rewardsPerEpisode[agent][states.episodeCount] += agentReward
                currState = self.agents[agent].state_to_device(statesForAgents[agent][0])

                # State, action, new state, reward, terminated (make sure this is always accurate)
                if prevState != None: # Skip first state, cus game is not loaded yet
                    
                    # All of these are already in tensors for further processing
                    experience = (prevState, self.lastAction[agent], currState, agentRewardTensor, states.isTerminated)
                    self.agents[agent].memory.append(experience)


                self.lastAction[agent] = self.agents[agent].action_to_device(self.action_to_idx(nextAgentAction))

                # change out the keys for that agent to the ones defined by an action
                keys = self.action_to_input(agent, nextAgentAction, keys) 
            else:
                keys = self.action_to_input(agent, rl_settings.ACTIONS[self.lastAction[agent]], keys) 
        return keys
    
    def post_episode_actions(self):

        if global_settings.DEBUG_MODE:
            print(f"Episode: {states.episodeCount}")
            print(f"p1 episode reward: {states.rewardsPerEpisode["player_one"][states.episodeCount - 1]}")
            print(f"p2 episode reward: {states.rewardsPerEpisode["player_two"][states.episodeCount - 1]}")
            print(f"Running epsilon: {states.epsilon}")

        for agentName in self.agents.keys():
            agent = self.agents[agentName]
            # If enough experience has been collected
            if len(agent.memory) > rl_settings.MINI_BATCH: 
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

