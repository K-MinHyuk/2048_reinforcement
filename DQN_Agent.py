import torch.nn as nn
from collections import deque
import numpy as np
import random
import torch

class DQNAgent:
    def __init__(self, state_size, action_size, load_model, model_path, render, discount_factor=0.999
                 , learning_rate=1e-4, epsilon=1.0, epsilon_decay=0.99999, epsilon_min = 0.01
                 , batch_size = 1024, train_start = 1024, queueLenMax = 5000):
        self.state_size = state_size
        self.action_size = action_size
        self.load_model = load_model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.train_loss = 0
        self.render = render
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epsilon_decay= epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.train_start = train_start
        self.queueLenMax = queueLenMax
        self.memory = deque(maxlen = self.queueLenMax)
        self.perfWindowSize =10
        self.penalty = -np.inf
        
    
        self.model ,self.model_optim, self.model_loss= self.build_model()
        self.target_model, self.target_model_optim, self.target_model_loss = self.build_model()
    
        self.update_target_model() # set same weights at the first time
        
        if self.load_model:
            self.model.load_state_dict(torch.load('./save_model/{}.pth'.format(model_path)))
            self.target_model.load_state_dict(torch.load('./save_model/{}.pth'.format(model_path)))
            self.epsilon_decay =1 
            self.epsilon = 0.01
            self.epsilon_min = 0.01
            print('Model_loaded...')

    def build_model(self):
        
        model = nn.Sequential(nn.Conv2d(1,16, kernel_size=2,stride=1),
                              nn.ReLU(),
                              nn.Conv2d(16,32, kernel_size=2,stride=1),
                              nn.ReLU(),
                              nn.Conv2d(32,64, kernel_size=2,stride=1),
                              nn.ReLU(),
                              nn.Flatten(),
                              nn.Linear(64,128),
                              nn.ReLU(),
                              nn.Linear(128,16),
                              nn.ReLU(),
                              nn.Linear(16,4))
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr = self.learning_rate)
        loss = nn.MSELoss()

        return model, optimizer, loss
    
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
        
    
    def get_action(self, state, env):
        if np.random.rand() <= self.epsilon:
            legal_moves = list()
            for i in range(4):
                temp_board = env.canMove(i, state)
                if(np.array_equal(temp_board,state)):
                    continue
                else:
                    legal_moves.append(i)
            return random.choice(legal_moves)
        else:
            state = torch.tensor([state], dtype =torch.float32).reshape(-1,1,4,4).to(self.device)
            state_ = state.clone()
            state_[state_ == 0] = 1.
            q_value = self.model(torch.log2(state_))
            q_value = q_value.cpu().detach().numpy()
            legal_moves = list()
            for i in range(4):
                state_, temp_board = env.canMove_t(i, state)
                if np.array_equal(temp_board,state_):
                    continue
                else:
                    legal_moves.append(i)
            while np.argmax(q_value[0]) not in legal_moves:
                q_value[0][np.argmax(q_value[0])] = -np.inf
                if len(legal_moves) == 0:
                    break
                continue
            return np.argmax(q_value[0])
        
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

            
    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min
            
        mini_batch = random.sample(self.memory, self.batch_size)
        states = np.zeros((self.batch_size, self.state_size))
        next_states = np.zeros((self.batch_size, self.state_size))
        actions, rewards, dones = [], [], []

        for i in range(self.batch_size):
            states[i] = mini_batch[i][0]
            actions.append(mini_batch[i][1])
            rewards.append(mini_batch[i][2])
            next_states[i] = mini_batch[i][3]
            dones.append(mini_batch[i][4])
    
        
        states = torch.tensor(states, dtype =torch.float32).reshape(-1,1,4,4).to(self.device)
        states[states==0] = 1.
        next_states = torch.tensor(next_states, dtype =torch.float32).reshape(-1,1,4,4).to(self.device)
        next_states[next_states==0] = 1.
        
        target = self.model(torch.log2(states))
        target_val = self.target_model(torch.log2(next_states))
    
        target_val = target_val.cpu().detach().numpy()
        
        for i in range(self.batch_size):
            if dones[i]:
                target[i][actions[i]] = self.penalty
            else:
                target[i][actions[i]] = rewards[i] + self.discount_factor * (np.amax(target_val[i]))

        # and do the model fit!
        self.model.train()
        self.model_optim.zero_grad()
        
        output = self.model(torch.log2(states))
        loss = self.model_loss(output, target)
        loss.backward()
        self.train_loss += loss.item()
        self.model_optim.step()
        
      