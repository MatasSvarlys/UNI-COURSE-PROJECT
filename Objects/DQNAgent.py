import torch
import torch.nn as nn
import torch.nn.functional as F
from Settings import global_settings

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
    def __init__(self, stateSize, action_size):
        self.state_size = stateSize
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # init dqn
        self.q_network = DQNetwork(stateSize, action_size).to(self.device)
        
        # TODO: make this map get created automatically from reading rl settings
        self.action_map = {
            0: "NOOP",
            1: "LEFT", 
            2: "RIGHT",
            3: "JUMP",
            4: "LEFT_JUMP",
            5: "RIGHT_JUMP"
        }
        
        if global_settings.DEBUG_MODE:
            print(f"DQNAgent initialized: state_size={stateSize}, action_size={action_size}")
            print(f"Using device: {self.device}")
            print(f"Network architecture: {self.q_network}")

    def step(self, state):
        # state is a np float array that gets turned into a torch acceptable tensor
        # the unsqueeze 0 creates a new dimention to represent batch size of 1
        # then it gets put into a device for calculations
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # pass the tensor into the network and get its calculated q values
            q_values = self.q_network(state_tensor)
            # then pick the highest evaluated one
            action_idx = torch.argmax(q_values, dim=1).item()
        
        # pick the action from the action map with default being noop
        action = self.action_map.get(action_idx, "NOOP")
        
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