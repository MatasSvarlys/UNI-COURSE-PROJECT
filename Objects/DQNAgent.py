import math

from PIL import Image

import numpy as np
import torch
# import torch_directml
import torch.nn as nn
import torch.nn.functional as F
from Objects.PrioritizedExperienceReplay import PrioritizedReplayMemory
from Settings import global_settings, rl_settings
from Objects.ExperienceReplay import ReplayMemory
from helper_functions.logger import log_q_values
from Objects import States

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initial noise level
        self.std_init = std_init

        # Learnable parameters (Means and Standard Deviations)
        # in a standard linear layer its y = weight * x + bias
        # here both are sampled from a normal distribution defined by these parameters
        # mu is the mean (the middle) and sigma is the standard deviation (the spread)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Random noise buffers (not learnable)
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        # the range is based of the size of the input, so it doesnt scale out of control
        mu_range = 1 / math.sqrt(self.in_features)
        # then the weights are initialized randomly in that range 
        # and the noise is initialized to a small value based on the size of the input
        # this is necessary to let the model understand which neuron is responsible for what
        # but keep the sigma (the spread) the same so the noise is consistent
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.weight_mu.device)
        # since the noise will be multiplied to create a matrix, we want the sqrt of it
        # but it can be negative, so we take the sign of it
        return x.sign().mul_(x.abs().sqrt())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        # here is the matrix from the 2 small vectors of noise
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if rl_settings.TRAINING_MODE:
            # the linear layer where the weight and bias is sampled from the normal distribution 
            return F.linear(x, 
                self.weight_mu + self.weight_sigma * self.weight_epsilon, 
                self.bias_mu + self.bias_sigma * self.bias_epsilon)
        else:
            # the standard linear layer
            return F.linear(x, self.weight_mu, self.bias_mu)

class DQNetwork(nn.Module):
    def __init__(self, input_channels, action_size):
        super(DQNetwork, self).__init__()
        # input_channels will be equal to STEPS_PER_ACTION as the colors are grayscaled
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Output from conv layers for 84x84 is 64 * 7 * 7 = 3136
        self.flatten_size = 64 * 7 * 7

        LinearLayer = NoisyLinear if rl_settings.USE_NOISY_NETS else nn.Linear

        if rl_settings.USE_DUELING_DQN:
            # Dueling Architecture
            self.value_stream = nn.Sequential(
                LinearLayer(self.flatten_size, 512),
                nn.ReLU(),
                LinearLayer(512, 1)
            )
            self.advantage_stream = nn.Sequential(
                LinearLayer(self.flatten_size, 512),
                nn.ReLU(),
                LinearLayer(512, action_size)
            )
        else:
            # Classic DQN Architecture
            self.fc1 = LinearLayer(self.flatten_size, 512)
            self.fc2 = LinearLayer(512, action_size)

    # goes through all the layers and resets noise for the noisy linear layers
    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, x):
        # x shape: [Batch, Channels, Height, Width]
        # print("x.mean(): ", x.mean())
        x = F.relu(self.conv1(x))
        # print("x.std(): ", x.std())
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten for the fully connected layer
        x = x.view(x.size(0), -1) 
        
        if rl_settings.USE_DUELING_DQN:
            # Combine Value and Advantage
            value = self.value_stream(x)
            advantage = self.advantage_stream(x)
            # Q = V + (A - mean(A))
            return value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            # Standard path
            x = F.relu(self.fc1(x))
            return self.fc2(x)
    

class DQNAgent:
    def __init__(self, action_size, isTraining, loss_logger, q_logger):
        self.action_size = action_size
        self.input_channels = rl_settings.STEPS_PER_ACTION
        self.loss_logger = loss_logger
        self.q_logger = q_logger
        self.loss_accumulator = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch_directml.device(torch_directml.default_device())
        
        self.isTraining = isTraining 

        if rl_settings.USE_PRIORITIZED_EXPERIENCE_REPLAY:
            self.memory = PrioritizedReplayMemory(rl_settings.MEMORY_SIZE)
        else:
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
        
        # state is a np uint8 array that gets turned into a torch acceptable tensor
        # the unsqueeze 0 creates a new dimention to represent batch size of 1
        # then it gets put into a device for calculations
        # print(f"step: {state.shape}")

        state_np = np.array(state)
        state_np = state_np.reshape(-1, rl_settings.IMAGE_HEIGHT, rl_settings.IMAGE_WIDTH)

        state_tensor = torch.from_numpy(state_np).float().to(self.device) / 255.0 
        state_tensor = state_tensor.unsqueeze(0)
        
        # debug_frames = state_np.astype(np.uint8)
        # side_by_side = np.hstack(debug_frames[:4])
    
        # Image.fromarray(side_by_side).save("debug_state.png")

        if rl_settings.USE_NOISY_NETS:
            self.policy_network.reset_noise()

        with torch.no_grad():
            # pass the tensor into the network and get its calculated q values
            q_values = self.policy_network(state_tensor)
            log_q_values(self.q_logger, States.episodeCount, States.episodeFrame, q_values.cpu().numpy()[0])
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
        
        if rl_settings.USE_NOISY_NETS:
            policy_dqn.reset_noise()
            target_dqn.reset_noise()

        if rl_settings.USE_PRIORITIZED_EXPERIENCE_REPLAY:
            # batch_data contains (samples, indices, is_weights)
            samples, indices, is_weights = mini_batch
            is_weights = is_weights.to(self.device)
        else:
            # batch_data is just the list of transitions
            samples = mini_batch
            indices = None
            is_weights = torch.ones(len(samples)).to(self.device) # Weight = 1 for standard DQN
        
        states, actions, newStates, rewards, terminations = zip(*samples)

        # States is of shape [mini_batch, frame_skip_steps, actual_size]
        # print(f"optimize: {np.array(states).shape}")
        # States has to be of shape [mini_batch, frame_skip_steps * actual_size]

        # 1. Convert to NumPy array (uint8)
        states_np = np.array(states)
        new_states_np = np.array(newStates)

        # 2. Move to Device as uint8
        states_t = torch.from_numpy(states_np).to(self.device)
        new_states_t = torch.from_numpy(new_states_np).to(self.device)

        # 3. Cast to float and Normalize on the GPU/Device
        states = states_t.float() / 255.0
        newStates = new_states_t.float() / 255.0

        # 4. Reshape
        # Assumes shape [Batch, Channels, Height, Width], -1 will be (frame_skip_steps * actual_channels)
        states = states.view(states.size(0), -1, rl_settings.IMAGE_HEIGHT, rl_settings.IMAGE_WIDTH)
        newStates = newStates.view(newStates.size(0), -1, rl_settings.IMAGE_HEIGHT, rl_settings.IMAGE_WIDTH)
        
        actions = torch.stack([torch.as_tensor(a, device=self.device) for a in actions]).long()
        rewards = torch.stack([torch.as_tensor(r, device=self.device) for r in rewards])
        terminations = torch.tensor(terminations, dtype=torch.float32, device=self.device)

        # --- Double DQN Logic ---
        with torch.no_grad():
            if rl_settings.USE_DOUBLE_DQN:
                bestActions = policy_dqn(newStates).argmax(dim=1).unsqueeze(dim=1)
                targetQ = rewards + (1 - terminations) * rl_settings.DISCOUNT_GAMA * target_dqn(newStates).gather(dim=1, index=bestActions).squeeze()
            else:
                targetQ = rewards + (1 - terminations) * rl_settings.DISCOUNT_GAMA * target_dqn(newStates).max(dim=1)[0]

        currQ = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

        td_errors = torch.abs(currQ - targetQ).detach().cpu().numpy()

        if rl_settings.USE_PRIORITIZED_EXPERIENCE_REPLAY and indices is not None:
            for i in range(len(indices)):
                self.memory.update(indices[i], td_errors[i])


        loss_unweighted = F.mse_loss(currQ, targetQ, reduction='none')
        loss = (loss_unweighted * is_weights).mean()        
        
        self.loss_logger.info(loss.item())
        
        # Optimize the model
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients
        
        # This has been used in the human-level-control-through-deep-reinforcement-learning paper
        # to make sure the model does not have major leaps in learning
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), max_norm=1.0)

        self.optimizer.step()       # Update network parameters
