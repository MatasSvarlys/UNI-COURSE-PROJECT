import json
import pygame
from Settings import rl_settings
from Objects.DQNAgent import DQNAgent

class AgentController:
    def __init__(self, stateSize):
        self.stateSize = stateSize
        self.agents = {}
        self.setup_agents()

    def setup_agents(self):
        if rl_settings.RL_CONTROL["player_one"]:
            self.agents["player_one"] = DQNAgent(stateSize=self.stateSize, action_size=rl_settings.ACTION_SPACE_SIZE)
        if rl_settings.RL_CONTROL["player_two"]:
            self.agents["player_two"] = DQNAgent(stateSize=self.stateSize, action_size=rl_settings.ACTION_SPACE_SIZE)
    
    def step_all_agents(self, statesForAgents, keys):

        
        keys = list(keys)

        for agent in self.agents.keys():
            # get the next action for the agent
            nextAgentAction = self.agents[agent].step(statesForAgents[agent]) 
            # change out the keys for that agent to the ones defined by an action
            keys = self.action_to_input(agent, nextAgentAction, keys) 

        return keys
        
    def action_to_input(self, player_name, action, keys):
        config_file = "./PlayerKeybinds/p1.json" if player_name == "player_one" else "./PlayerKeybinds/p2.json"

        # Load key mappings from JSON file
        with open(config_file, 'r') as f:
            key_mappings = json.load(f)
        
        # Block all keys defined in the config file
        for _, key_name in key_mappings.items():
            key_code = getattr(pygame, key_name)
            keys[key_code] = 0

        match action:
            case "NOOP":
                pass
            case "LEFT":
                keys[getattr(pygame, key_mappings["MOVE_LEFT"])] = 1
            case "RIGHT":
                keys[getattr(pygame, key_mappings["MOVE_RIGHT"])] = 1
            case "JUMP":
                keys[getattr(pygame, key_mappings["JUMP"])] = 1
            case "LEFT_JUMP":
                keys[getattr(pygame, key_mappings["MOVE_LEFT"])] = 1
                keys[getattr(pygame, key_mappings["JUMP"])] = 1
            case "RIGHT_JUMP":
                keys[getattr(pygame, key_mappings["MOVE_RIGHT"])] = 1
                keys[getattr(pygame, key_mappings["JUMP"])] = 1

        return keys

