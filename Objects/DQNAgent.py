import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from Settings import global_settings, rl_settings
from Objects.ExperienceReplay import ReplayMemory

class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.input_size = state_size * rl_settings.FRAME_SKIPPING_STEPS
        self.fc1 = nn.Linear(self.input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
        # print(f"forward: {x.shape}")
        batch_size = x.shape[0]
        # print(batch_size.size)
        x = x.view(batch_size, -1)
        # print(f"forward2: {x.shape}")

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent:
    def __init__(self, stateSize, action_size, isTraining):
        self.state_size = stateSize
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.isTraining = isTraining 
        self.memory = ReplayMemory(rl_settings.MEMORY_SIZE)

        self.prevState = None
        
        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        # init dqn
        self.policy_network = DQNetwork(stateSize, action_size).to(self.device)

        if isTraining:
            self.target_network = DQNetwork(stateSize, action_size).to(self.device)
            # Make sure the initial weights are the same
            self.target_network.load_state_dict(self.policy_network.state_dict())
            self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=rl_settings.LEARNING_RATE)

        self.action_map = rl_settings.ACTIONS
        
        if global_settings.DEBUG_MODE:
            print(f"DQNAgent initialized: state_size={stateSize}, action_size={action_size}")
            print(f"Using device: {self.device}")
            print(f"Network architecture: {self.policy_network}")


    def float_list_to_device(self, state):
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)

    def float_to_device(self, action):
        return torch.tensor(action, dtype=torch.long, device=self.device)


    def step(self, state):
        
        # state is a np float array that gets turned into a torch acceptable tensor
        # the unsqueeze 0 creates a new dimention to represent batch size of 1
        # then it gets put into a device for calculations
        # print(f"step: {state.shape}")

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # pass the tensor into the network and get its calculated q values
            q_values = self.policy_network(state_tensor)

            # then pick the highest evaluated one
            action_idx = torch.argmax(q_values, dim=1).item()
        
        # translate to the action from the action map 
        action = self.action_map[action_idx]
        
        # TODO: i should probably make a debug object atp
        if hasattr(self, 'debug_counter'):
            self.debug_counter += 1
        else:
            self.debug_counter = 0
            
        if global_settings.DEBUG_MODE and self.debug_counter % 60 == 0:
            print(f"Agent state shape: {state.shape}")
            print(f"Q-values: {q_values.cpu().numpy()[0]}")
            print(f"Selected action: {action} (index: {action_idx})")
        
        return action
    
    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        states, actions, newStates, rewards, terminations = zip(*mini_batch)

        # States is of shape [mini_batch, frame_skip_steps, actual_size]
        # print(f"optimize: {np.array(states).shape}")

        # States has to be of shape [mini_batch, frame_skip_steps * actual_size]
        states = torch.from_numpy(np.array(states)).float().to(self.device)
        states = states.view(states.size(0), -1)
        # print(f"optimize: {states.shape}")

        actions = torch.stack(actions)
        newStates = torch.from_numpy(np.array(newStates)).float().to(self.device)
        newStates = newStates.view(newStates.size(0), -1)
        rewards = torch.stack(rewards)

        terminations = torch.tensor(terminations).float().to(self.device)

        with torch.no_grad():
            targetQ = rewards + (1-terminations) * rl_settings.DISCOUNT_GAMA * target_dqn(newStates).max(dim=1)[0]


        currQ = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        if global_settings.DEBUG_MODE:
            print(f"Current Q values: {currQ} -> target Q values: {targetQ}")
        
        loss = self.loss_fn(currQ, targetQ)

        # Optimize the model
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients
        
        # This has been used in the human-level-control-through-deep-reinforcement-learning paper
        # to make sure the model does not have major leaps in learning
        for param in policy_dqn.parameters():
            param.grad.data.clamp_(-1, 1)

        self.optimizer.step()       # Update network parameters
