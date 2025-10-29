import torch
import torch.nn as nn
import torch.nn.functional as F
from Settings import global_settings, rl_settings
from Objects.ExperienceReplay import ReplayMemory

class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, action_size)
        
    def forward(self, x):
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

    def step(self, state):

        self.prevState = torch.FloatTensor(state).unsqueeze(0)

        # state is a np float array that gets turned into a torch acceptable tensor
        # the unsqueeze 0 creates a new dimention to represent batch size of 1
        # then it gets put into a device for calculations
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # pass the tensor into the network and get its calculated q values
            q_values = self.policy_network(state_tensor)
            # then pick the highest evaluated one
            action_idx = torch.argmax(q_values, dim=1).item()
        
        # pick the action from the action map with default being noop
        action = self.action_map[action_idx]
        
        # TODO: i should probably make a debug object atp
        if hasattr(self, 'debug_counter'):
            self.debug_counter += 1
        else:
            self.debug_counter = 0
            
        if self.debug_counter % 60 == 0:
            print(f"Agent state shape: {state.shape}")
            print(f"Q-values: {q_values.cpu().numpy()[0]}")
            print(f"Selected action: {action} (index: {action_idx})")
        
        return action
    
    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        for state, action, new_state, reward, terminated in mini_batch:
            new_state.to(self.device)
            state.to(self.device)
            if terminated:
                target = reward
            else:
                with torch.no_grad():
                    targetQ = reward + rl_settings.DISCOUNT_GAMA * target_dqn(new_state).max()

        currQ = policy_dqn(state)

        loss = self.loss_fn(currQ, targetQ)
        # Optimize the model (backpropagation)
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients
        self.optimizer.step()       # Update network parameters i.e. weights and biases
