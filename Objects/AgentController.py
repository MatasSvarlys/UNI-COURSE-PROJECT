from collections import deque
import logging
import numpy as np
import torch
from Settings import rl_settings
from Settings import global_settings
from Objects.DQNAgent import DQNAgent
import random
import os
import Objects.States as states

class AgentController:
    def __init__(self, stateSize):
        self.stateSize = stateSize

        self.agentNames = [k for k, v in rl_settings.RL_CONTROL.items() if v]
        self.agents = {}
        self.loggers = {}
        self.isTraining = rl_settings.TRAINING_MODE

        self.frameHistory = {}
        self.episodeRewards = {}
        self.lastAction = {}
        
        self.randFrames = random.randrange(1, rl_settings.FRAMES_PER_STEP)
        self.rlFrames = 0

        self.setup_logging()
        self.setup_agents()

        if rl_settings.LOAD_MODEL:
            self.load_agents("loads")
    
    def setup_logging(self):
        os.makedirs("logs", exist_ok=True)
        
        for agentName in self.agentNames:
            logger = logging.getLogger(agentName)
            logger.setLevel(logging.INFO)
            
            # make logs not appear in console
            logger.propagate = False 
            
            # Create file handler
            log_file = os.path.join("logs", f"{agentName}_log.csv")
            file_handler = logging.FileHandler(log_file, mode='a')
            
            # CSV formatting: Time, Episode, Epsilon, Random?, Action, RunningReward
            formatter = logging.Formatter('%(message)s')
            file_handler.setFormatter(formatter)
            
            if not logger.handlers:
                logger.addHandler(file_handler)
                # Write header if file is new
                if os.stat(log_file).st_size == 0:
                    logger.info("Episode,Epsilon,IsRandom,Action,RunningReward")
            
            self.loggers[agentName] = logger


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
            self.frameHistory[agentName] = deque(maxlen=rl_settings.STEPS_PER_ACTION * 2) # one for current and one for past state blocks
            self.episodeRewards[agentName] = {}
            self.lastAction[agentName] = 0

    def step_all_agents(self, statesForAgents):
        self.currEpsilon = states.epsilon
        states.epsilon = max(states.epsilon - rl_settings.EPSILON_DECAY, rl_settings.MIN_EPSILON)
        nextActions = {}

        # if states.isTerminated:
        #     self.post_episode_actions()

        # Only act once every n frames with a random offset
        if (states.episodeFrame + self.randFrames) % rl_settings.FRAMES_PER_STEP == 0:
            self.rlFrames += 1
            
            for agentName in self.agentNames:
                nextActions[agentName] = self.step_one_agent(agentName, statesForAgents[agentName])

        # All other frames repeat the last action
        else:
            for agentName in self.agentNames:
                nextActions[agentName] = self.lastAction[agentName]
            

        # Every few steps update the models   
        if rl_settings.TRAINING_MODE and states.episodeCount > rl_settings.EXPERIENCE_COLLECTION_EPISODES and self.rlFrames % rl_settings.NETWORK_LEARN_RATE == 0:
            self.update_models()

        
        return nextActions
    
    def update_models(self):
        for agentName in self.agentNames:
            agent = self.agents[agentName]
            # If enough experience has been collected
            if len(agent.memory) > rl_settings.MINI_BATCH: 
                mini_batch = agent.memory.sample(rl_settings.MINI_BATCH)
                # Optimize the policy network
                agent.optimize(mini_batch, agent.policy_network, agent.target_network)
        
        pass

    def step_one_agent(self, agentName, stateObject):

        # agentState = stateObject[0]


        # If it's the first few steps of the episode, take random actions untill the frame history fills up. 
        if self.rlFrames <= rl_settings.STEPS_PER_ACTION * 2:
            nextAgentAction = self.pick_random_action()
            self.lastAction[agentName] = self.agents[agentName].float_to_device(nextAgentAction)

            is_random = True
            self.log_action(agentName, is_random, nextAgentAction, states.episodeReward[agentName])
            return nextAgentAction



        # len(self.frameHistory[agentName]) has to be 2*rl_settings.STEPS_PER_ACTION
        recent_frames = list(self.frameHistory[agentName])

        # lastStackedState = np.stack(recent_frames[:rl_settings.STEPS_PER_ACTION])
        currStackedState = np.stack(recent_frames[rl_settings.STEPS_PER_ACTION:])

        # If not training, just get the action and move on
        is_random = False
        if not rl_settings.TRAINING_MODE:
            nextAgentAction = self.agents[agentName].step(currStackedState) 
            self.lastAction[agentName] = nextAgentAction
            return nextAgentAction

        # If the game was terminated, do nothing
        if states.isTerminated:
            nextAgentAction = 0
            self.lastAction[agentName] = nextAgentAction
            return nextAgentAction
            
        # If it's the first few episodes, collect dummy data to lessen the overfitting 
        # to the begginging of learning process
        if states.episodeCount < rl_settings.EXPERIENCE_COLLECTION_EPISODES:
            nextAgentAction = self.pick_random_action()
            self.lastAction[agentName] = self.agents[agentName].float_to_device(nextAgentAction)

            is_random = True
            self.log_action(agentName, is_random, nextAgentAction, states.episodeReward[agentName])
            return nextAgentAction

        # After that, chose the action in an epsilon greedy fashion
        # TODO: Make the random choice seeded for replication
        is_random = False
        if random.random() < self.currEpsilon:
            nextAgentAction = self.pick_random_action()
            is_random = True
        else:
            nextAgentAction = self.agents[agentName].step(currStackedState) 

        self.log_action(agentName, is_random, nextAgentAction, states.episodeReward[agentName])
        
        self.lastAction[agentName] = self.agents[agentName].float_to_device(nextAgentAction)

        return nextAgentAction


    def post_episode_actions(self):
        
        if states.episodeCount % 10000 == 0 and rl_settings.TRAINING_MODE:
            self.save_agents("saves", states.episodeCount)

        for agentName in self.agentNames:
            self.log_episode_end(agentName, states.episodeReward[agentName])
            
            agent = self.agents[agentName]

            if states.episodeCount % rl_settings.NETWORK_SYNC_RATE == 0 and len(agent.memory) > rl_settings.MINI_BATCH and rl_settings.TRAINING_MODE and states.episodeCount > rl_settings.EXPERIENCE_COLLECTION_EPISODES: 
                    self.total_steps = 0    
                    agent.target_network.load_state_dict(agent.policy_network.state_dict())


        self.randFrames = random.randrange(1, rl_settings.FRAMES_PER_STEP)


    def save_experience(self, agentName, lastState, action, currentState, reward, terminated):
        if (states.episodeFrame + self.randFrames) % rl_settings.FRAMES_PER_STEP != 0:
            return

        agent = self.agents[agentName]
        
        action_tensor = agent.float_to_device(action)
        reward_tensor = agent.float_to_device(reward)
        
        experience = (
            lastState, 
            action_tensor, 
            currentState, 
            reward_tensor, 
            terminated
        )
        
        agent.memory.append(experience)

    def pick_random_action(self):
        return random.randint(0, len(rl_settings.ACTIONS) - 1)

    def log_action(self, agentName, is_random, action, reward):
        # Format: Episode, Epsilon, IsRandom, Action, RunningReward
        log_msg = f"episode: {states.episodeCount}, frame: {states.episodeFrame}, epsilon: {states.epsilon:.10f}, random: {is_random}, action: {rl_settings.ACTIONS[action]}, reward: {reward:.2f}"
        self.loggers[agentName].info(log_msg)

    def log_episode_end(self, agentName, total_reward):
        log_msg = f"SUMMARY for ep {states.episodeCount}: total reward = {total_reward:.2f}"
        
        # We use a visual separator in the log to make it easy to scan
        self.loggers[agentName].info(log_msg)