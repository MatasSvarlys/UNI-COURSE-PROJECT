import numpy as np
import torch
# import torch_directml
import torch.nn as nn
import torch.nn.functional as F
from Settings import global_settings, rl_settings
from Objects.ExperienceReplay import ReplayMemory

class DQNetwork(nn.Module):
    def __init__(self, input_channels, action_size):
        super(DQNetwork, self).__init__()
        # input_channels will be (3 * STEPS_PER_ACTION)
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # We need to calculate the size of the output from conv layers to feed into FC
        # For an 84x84 input, the output of these specific convs is 7x7x64
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_size)
        
    def forward(self, x):
        # x shape: [Batch, Channels, Height, Width]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten for the fully connected layer
        x = x.view(x.size(0), -1) 
        
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

class DQNAgent:
    def __init__(self, action_size, isTraining):
        self.action_size = action_size
        self.input_channels = 3 * rl_settings.STEPS_PER_ACTION

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch_directml.device(torch_directml.default_device())
        
        self.isTraining = isTraining 
        self.memory = ReplayMemory(rl_settings.MEMORY_SIZE)

        self.prevState = None
        
        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        # init dqn
        self.policy_network = DQNetwork(self.input_channels, action_size).to(self.device)

        if isTraining:
            self.target_network = DQNetwork(self.input_channels, action_size).to(self.device)
            # Make sure the initial weights are the same
            self.target_network.load_state_dict(self.policy_network.state_dict())
            self.optimizer = torch.optim.RMSprop(self.policy_network.parameters(), lr=rl_settings.LEARNING_RATE)

        self.action_map = rl_settings.ACTIONS
        
        if global_settings.DEBUG_MODE:
            print(f"DQNAgent initialized: state_size={self.input_channels}, action_size={action_size}")
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

        state_np = np.array(state)
        state_np = state_np.reshape(-1, rl_settings.IMAGE_HEIGHT, rl_settings.IMAGE_WIDTH)
        state_tensor = torch.from_numpy(state_np).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # pass the tensor into the network and get its calculated q values
            q_values = self.policy_network(state_tensor)
            # print(q_values)
            with open('q_values_log.txt', 'a') as f:
                f.write(f"{q_values.cpu().numpy()[0]}\n")
            # then pick the highest evaluated one
            action_idx = torch.argmax(q_values, dim=1).item()

        # TODO: i should probably make a debug object atp
        if hasattr(self, 'debug_counter'):
            self.debug_counter += 1
        else:
            self.debug_counter = 0
            
        if global_settings.DEBUG_MODE and self.debug_counter % 60 == 0:
            print(f"Agent state shape: {state.shape}")
            print(f"Q-values: {q_values.cpu().numpy()[0]}")
            print(f"Selected action: {rl_settings.ACTIONS[action_idx]} (index: {action_idx})")
        
        return action_idx
    
    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        states, actions, newStates, rewards, terminations = zip(*mini_batch)

        # States is of shape [mini_batch, frame_skip_steps, actual_size]
        # print(f"optimize: {np.array(states).shape}")
        # States has to be of shape [mini_batch, frame_skip_steps * actual_size]
        states = torch.from_numpy(np.array(states)).float().to(self.device)
        states = states.reshape(states.shape[0], -1, rl_settings.IMAGE_HEIGHT, rl_settings.IMAGE_WIDTH)
        newStates = torch.from_numpy(np.array(newStates)).float().to(self.device)        
        newStates = newStates.reshape(newStates.shape[0], -1, rl_settings.IMAGE_HEIGHT, rl_settings.IMAGE_WIDTH)

        actions = torch.stack(actions)
        rewards = torch.stack(rewards)

        terminations = torch.tensor(terminations).float().to(self.device)

        with torch.no_grad():
            # TODO: make this a setting

            doubleDQN = True

            if doubleDQN:
                bestActions = policy_dqn(newStates).argmax(dim=1)

                targetQ = rewards + (1-terminations) * rl_settings.DISCOUNT_GAMA * target_dqn(newStates).gather(dim=1, index=bestActions.unsqueeze(dim=1)).squeeze()

            else:
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
