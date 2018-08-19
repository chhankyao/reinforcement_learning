import gym
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# ===== CartPole environment =====
env = gym.make('CartPole-v0')
env._max_episode_steps = 500
env.reset()



# ===== DQN network =====
class QNetwork(nn.Module):      
    def __init__(self, learning_rate, state_size, action_size, hidden_size, alpha_decay):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.fc2.weight.data.normal_(0, 0.1)
        
        self.optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optim, step_size=50, gamma=alpha_decay)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = Variable(torch.Tensor(x))
        x = self.fc1(x)
        x = F.tanh(x)
        actions_value = self.fc2(x)
        return actions_value
    
    def update(self, states, qvalues):
        self.train(mode=True)
        self.optim.zero_grad()
        Q_pred = self.forward(states)
        Q_true = Variable(torch.Tensor(qvalues))
        loss = self.loss_fn(Q_pred, Q_true)
        loss.backward()
        self.optim.step()
        self.lr_scheduler.step()
        self.train(mode=False)

        
# ===== Experience replay buffer =====
class Replay():       
    def __init__(self, max_size):
        self.capacity = max_size
        self.memory = []
        self.position = 0
        
    def initialize(self, init_length, envir):
        self.position = init_length
        s = envir.reset()
        for i in range(init_length):
            a = envir.action_space.sample()
            next_s, r, done, _ = envir.step(a)
            self.memory.append(None)
            self.memory[i] = [s, a, r, next_s, done]
            if done:
                s = envir.reset()
            else:
                s = next_s

    def push(self, trans):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = trans
        self.position = (self.position + 1) % self.capacity
            
    def generate_minibatch(self, DQN, targetDQN, batch_size):
        def get_targetQ(trans):
            [s, a, r, next_s, done] = trans
            if done:
                targetQ = r
            else:
                targetQ = r + gamma * np.amax(targetDQN.forward(next_s).data.numpy())
            if a == 0:
                return np.array([targetQ, DQN.forward(s).data.numpy()[1]])
            else:
                return np.array([DQN.forward(s).data.numpy()[0], targetQ])
            
        trans = random.sample(self.memory, batch_size)
        states = np.vstack([x[0] for x in trans])
        Q_target = np.array([get_targetQ(x) for x in trans])
        return states, Q_target

    
    
# Initialize DQN
learning_rate = 0.1
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
hidden_size = 16
alpha_decay = 0.5

DQN = QNetwork(learning_rate, state_size, action_size, hidden_size, alpha_decay)
targetDQN = QNetwork(learning_rate, state_size, action_size, hidden_size, alpha_decay)

# set targetDQN weights to DQN weights
targetDQN.load_state_dict(DQN.state_dict())


# Populate the initial experience buffer
replay = Replay(max_size=10000)
replay.initialize(init_length=1000, envir=env)



# Runtime parameters
num_episodes = 10000           # max number of episodes to learn from
gamma = 0.99                   # future reward discount
max_steps = 500                # cut off simulation after this many steps
batch_size = 128

# Exploration parameters
min_epsilon = 0.01             # minimum exploration probability
decay_rate = 5/num_episodes    # exponential decay rate for exploration prob
returns = np.zeros(num_episodes)
step_list = np.zeros(num_episodes)


def get_action(state, epsilon):
    if np.random.uniform() > epsilon:
        return np.argmax(DQN.forward(state).data.numpy())
    else:
        return env.action_space.sample()

n_steps = 0

for ep in range(1, num_episodes):
    epsilon = min_epsilon + (1.0 - min_epsilon)*np.exp(-decay_rate*ep)
    
    # --> start episode
    s = env.reset()
    total_reward = 0
    
    done = False
    while not done:
        # explore/exploit and get action using DQN
        a = get_action(s, epsilon)
        
        # perform action and record new_state, action, reward
        next_s, r, done, _ = env.step(a)
        n_steps += 1
        
        # populate Replay experience buffer
        replay.push([s, a, r, next_s, done])
        
        total_reward += r
        s = next_s   
    # <-- end episode
    
    returns[ep] = total_reward
    step_list[ep] = n_steps
    
    # Replay
    states, qvalues = replay.generate_minibatch(DQN, targetDQN, batch_size)
    
    # set targetDQN weights to DQN weights
    targetDQN.load_state_dict(DQN.state_dict())
    
    # update DQN (run one epoch of training per episode with generated minibatch of states and qvalues)
    DQN.update(states, qvalues)

    
    
# plot average returns
returns_over_100_episodes = []
x = []
for i in range(0,int(num_episodes/100)):
    returns_over_100_episodes.append(sum(returns[100*i:100*(i+1)-1])/100)
    x.append((i+1)*100)
plt.plot(x,returns_over_100_episodes,'.-r')
plt.ylabel('Average Returns per Episode')
plt.xlabel('Num of Episodes')
plt.show()



# Demo final network
env.reset()

# Take one random step to get the pole and cart moving
state, reward, done, _ = env.step(env.action_space.sample())
# state = np.reshape(state, [1, state.size])
total_reward = 0

for i in range(0, max_steps):
    env.render()
    
    # Get action from Q-network
    # Qs = output of DQN.model when state is passed in
    Qs = DQN.forward(state).data.numpy()
    action = np.argmax(Qs)
    
    # Take action, get new state and reward
    next_state, reward, done, _ = env.step(action)
    total_reward += reward

    if done:
        break
    else:
        state = next_state
