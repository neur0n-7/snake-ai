"""

  ________  _____  ___        __       __   ___  _______           __        __     
 /"       )(\"   \|"  \      /""\     |/"| /  ")/"     "|         /""\      |" \    
(:   \___/ |.\\   \    |    /    \    (: |/   /(: ______)        /    \     ||  |   
 \___  \   |: \.   \\  |   /' /\  \   |    __/  \/    |         /' /\  \    |:  |   
  __/  \\  |.  \    \. |  //  __'  \  (// _  \  // ___)_       //  __'  \   |.  |   
 /" \   :) |    \    \ | /   /  \\  \ |: | \  \(:      "|     /   /  \\  \  /\  |\  
(_______/   \___|\____\)(___/    \___)(__|  \__)\_______)    (___/    \___)(__\_|_) 

Anish Gupta
https://github.com/neur0n-7
      
Deep Q-Learning agent class
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random

# DQN CLASS ##############################################################

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    


# AGENT CLASS ############################################################

class Agent:
    def __init__(self, input_dim, output_dim):
        self.model = DQN(input_dim, output_dim)
        self.replay_buffer = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    def act(self, state_vector):
        if random.random() < self.epsilon:
            # 0, 1, 2 is straight left right
            return random.randint(0, 2)
        with torch.no_grad():
            state_tensor = torch.tensor(state_vector, dtype=torch.float32)
            q_values = self.model(state_tensor)
            return torch.argmax(q_values).item()
        
    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train_from_buffer(self, batch_size=64):
        # not enough data yet
        if len(self.replay_buffer) < batch_size:
            return 

        batch = random.sample(self.replay_buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # turn into tensors
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # calc q values
        q_values = self.model(states) # [batch_size, 3]

        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1) # [batch_size]

        # Target Q vals
        with torch.no_grad():
            next_q_values = self.model(next_states)
            max_next_q_values = next_q_values.max(1)[0]
            targets = rewards + self.gamma * max_next_q_values * (1 - dones)

        loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        torch.save(self.model.state_dict(), path)