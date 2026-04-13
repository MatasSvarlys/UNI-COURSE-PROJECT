from collections import deque
from logging.handlers import QueueHandler

import logging
from torch import multiprocessing
from helper_functions.logger import logging_worker, log_action, log_episode_end
import torch
from Settings import global_settings, rl_settings
from Objects.DQNAgent import DQNAgent
import random
import os
import Objects.States as states

class AgentController:
    def __init__(self):

        self.agentNames = [k for k, v in rl_settings.RL_CONTROL.items() if v]
        self.agents = {}
        self.loggers = {}
        self.isTraining = rl_settings.TRAINING_MODE

        self.frameHistory = {}
        self.stackedState = {}
        self.lastStackedState = {}
        self.episodeRewards = {}
        self.lastAction = {}
        
        self.randFrames = random.randrange(1, rl_settings.FRAMES_PER_STEP)
        self.episodeStep = 0

        self.setup_logging()
        self.setup_agents()

        if rl_settings.LOAD_MODEL:
            self.load_agents("loads")

    def setup_logging(self):
        os.makedirs("logs", exist_ok=True)
        self.log_processes = []

        for agentName in self.agentNames:
            logger = logging.getLogger(agentName)
            logger.setLevel(logging.INFO)
            logger.propagate = False 

            agent_queue = multiprocessing.Queue(global_settings.LOG_QUEUE_SIZE)
            queue_handler = QueueHandler(agent_queue)
            logger.addHandler(queue_handler)

            log_file = os.path.join("logs", f"{agentName}_log.csv")
            
            # target now points to the function imported from LoggerUtils
            p = multiprocessing.Process(
                target=logging_worker, 
                args=(agent_queue, log_file),
                name=f"Logger-{agentName}",
                daemon=True
            )
            p.start()
            self.log_processes.append(p)
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
            self.agents[agentName] = DQNAgent(action_size=rl_settings.ACTION_SPACE_SIZE, isTraining=self.isTraining)
            self.frameHistory[agentName] = deque(maxlen=rl_settings.STEPS_PER_ACTION)
            self.stackedState[agentName] = None
            self.lastStackedState[agentName] = None
            self.episodeRewards[agentName] = {}
            self.lastAction[agentName] = 0

    def step_all_agents(self, statesForAgents):
        nextAction = {}

        # Since this is called every frame, we can save those frames in history here
        for agentName in self.agentNames:
            if agentName in statesForAgents:
                self.frameHistory[agentName].append(statesForAgents[agentName])

        # Only act once every n frames with a random offset
        if (states.episodeFrame + self.randFrames) % rl_settings.FRAMES_PER_STEP == 0:
            self.episodeStep += 1
            
            for agentName in self.agentNames:
                self.stackedState[agentName] = list(self.frameHistory[agentName])
                nextAction[agentName] = self.step_one_agent(agentName, self.stackedState[agentName])

        # All other frames repeat the last action
        else:
            for agentName in self.agentNames:
                nextAction[agentName] = self.lastAction[agentName]
            

        # Every few steps update the models   
        if rl_settings.TRAINING_MODE and states.episodeCount > rl_settings.EXPERIENCE_COLLECTION_EPISODES and self.episodeStep % rl_settings.NETWORK_LEARN_RATE == 0:
            self.update_models()

        
        return nextAction
    
    def update_models(self):
        for agentName in self.agentNames:
            agent = self.agents[agentName]
            # If enough experience has been collected
            if len(agent.memory) > rl_settings.MINI_BATCH: 
                mini_batch = agent.memory.sample(rl_settings.MINI_BATCH)
                # Optimize the policy network
                agent.optimize(mini_batch, agent.policy_network, agent.target_network)
        
        pass

    def step_one_agent(self, agentName, stackedState):

        # If not training, just get the action and move on
        isRandom = False
        if not rl_settings.TRAINING_MODE:
            nextAgentAction = self.agents[agentName].step(stackedState) 
            self.lastAction[agentName] = nextAgentAction
            
            log_action(
                self.loggers[agentName], 
                agentName, 
                states.episodeCount, 
                states.episodeFrame, 
                states.epsilon, 
                isRandom, 
                rl_settings.ACTIONS[nextAgentAction], 
                states.episodeReward[agentName]
            )
            
            return nextAgentAction

        # If the game was terminated, do nothing
        if states.isTerminated:
            nextAgentAction = 0
            self.lastAction[agentName] = nextAgentAction

            log_action(
                self.loggers[agentName], 
                agentName, 
                states.episodeCount, 
                states.episodeFrame, 
                states.epsilon, 
                isRandom, 
                rl_settings.ACTIONS[nextAgentAction], 
                states.episodeReward[agentName]
            )

            return nextAgentAction
            
        # If it's the first few episodes, collect dummy data to lessen the overfitting 
        # to the begginging of learning process
        if states.episodeCount < rl_settings.EXPERIENCE_COLLECTION_EPISODES:
            nextAgentAction = self.pick_random_action()
            self.lastAction[agentName] = nextAgentAction

            isRandom = True

            log_action(
                self.loggers[agentName], 
                agentName, 
                states.episodeCount, 
                states.episodeFrame, 
                states.epsilon, 
                isRandom, 
                rl_settings.ACTIONS[nextAgentAction], 
                states.episodeReward[agentName]
            )

            return nextAgentAction

        # After that, chose the action in an epsilon greedy fashion
        # TODO: Make the random choice seeded for replication
        isRandom = False
        if random.random() < states.epsilon:
            nextAgentAction = self.pick_random_action()
            isRandom = True
        else:
            nextAgentAction = self.agents[agentName].step(stackedState) 

        log_action(
            self.loggers[agentName], 
            agentName, 
            states.episodeCount, 
            states.episodeFrame, 
            states.epsilon, 
            isRandom, 
            rl_settings.ACTIONS[nextAgentAction], 
            states.episodeReward[agentName]
        )

        self.lastAction[agentName] = nextAgentAction

        return nextAgentAction


    def post_episode_actions(self):
        
        if states.episodeCount % 10000 == 0 and rl_settings.TRAINING_MODE:
            self.save_agents("saves", states.episodeCount)

        for agentName in self.agentNames:
            log_episode_end(
                self.loggers[agentName], 
                agentName, 
                states.episodeCount, 
                states.episodeReward[agentName]
            )


            agent = self.agents[agentName]
            if states.episodeCount % rl_settings.NETWORK_SYNC_RATE == 0 and len(agent.memory) > rl_settings.MINI_BATCH and rl_settings.TRAINING_MODE and states.episodeCount > rl_settings.EXPERIENCE_COLLECTION_EPISODES: 
                self.total_steps = 0    
                agent.target_network.load_state_dict(agent.policy_network.state_dict())


        self.randFrames = random.randrange(1, rl_settings.FRAMES_PER_STEP)


    def save_experience(self, agentName, lastState, action, currentState, reward, terminated):

        agent = self.agents[agentName]
        
        # action_tensor = agent.float_to_device(action)
        # reward_tensor = agent.float_to_device(reward)
        
        experience = (
            lastState, 
            # action_tensor, 
            action,
            currentState, 
            # reward_tensor,
            reward, 
            terminated
        )
        
        agent.memory.append(experience)

    def pick_random_action(self):
        return random.randint(0, len(rl_settings.ACTIONS) - 1)