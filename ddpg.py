import matplotlib.pyplot as plt
import numpy as np
import random
import os
import time

# environment
import gym

# pytorch
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

logging_interval = 100
animate_interval = logging_interval * 5
logdir='./DDPG/test/'

# make variable types for automatic setting to GPU or CPU, depending on GPU availability
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor



VISUALIZE = False
SEED = 0
MAX_PATH_LENGTH = 500
NUM_EPISODES = 2000
GAMMA = 0.99
BATCH_SIZE = 128

# environments to be tested on
env_name = 'InvertedPendulum-v1'
# env_name = 'Pendulum-v0'
# env_name = 'HalfCheetah-v1' 

env = gym.make(env_name)
env._max_episode_steps = MAX_PATH_LENGTH

# wrap gym to save videos
if VISUALIZE:
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    env = gym.wrappers.Monitor(env, logdir, force=True, video_callable=lambda episode_id: episode_id%animate_interval==0)

# check observation and action space
discrete = isinstance(env.action_space, gym.spaces.Discrete)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n if discrete else env.action_space.shape[0]
    
# set random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)



class NormalizeAction(gym.ActionWrapper):
    def __init__(self, env=None):
        super(NormalizeAction, self).__init__(env)
        
    def _action(self, action):
        # tanh outputs (-1,1) from tanh, need to be [action_space.low, action_space.high]
        r = (self.action_space.high - self.action_space.low) / 2.
        mid = (self.action_space.high + self.action_space.low) / 2.
        return mid + r * action
    
    def _reverse_action(self, action):
        # reverse of that above
        r_inv = 2. / (self.action_space.high - self.action_space.low)
        mid = (self.action_space.high + self.action_space.low) / 2.
        return r_inv * (action - mid)
    


def weightSync(target_model, source_model, tau = 0.001):
    for parameter_target, parameter_source in zip(target_model.parameters(), source_model.parameters()):
        parameter_target.data.copy_((1 - tau) * parameter_target.data + tau * parameter_source.data)
        
        

# replay buffer
class Replay():       
    def __init__(self, max_size=60000, init_length=1000):
        self.capacity = max_size
        self.memory = []
        self.position = init_length
        s = env.reset()
        for i in range(init_length):
            a = np.random.uniform(-1, 1, act_dim)
            next_s, r, done, _ = env.step(a)
            self.memory.append([s, a, r, next_s, done])
            if done:
                s = env.reset()
            else:
                s = next_s

    def push(self, trans):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = trans
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size=BATCH_SIZE):
        trans = np.asarray(random.sample(self.memory, batch_size))
        s = np.vstack(trans[:,0])
        a = np.vstack(trans[:,1])
        r = np.vstack(trans[:,2])
        next_s = np.vstack(trans[:,3])
        done = np.vstack(trans[:,4])
        return s, a, r, next_s, done
    
    

# random process for exploration
class OrnsteinUhlenbeckProcess():
    def __init__(self, dimension, num_steps, theta=0.15, mu=0., sigma=0.3, dt=0.01):
        super(OrnsteinUhlenbeckProcess, self).__init__()
        self.dimension = dimension
        self.num_steps = num_steps
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.reset()

    def step(self):
        dx = self.theta * (self.mu-self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt)* np.random.randn(self.dimension)
        self.x_prev = self.x_prev + dx
        return self.x_prev

    def reset(self):
        self.x_prev = np.ones(self.dimension) * self.mu
        
        
        
# ----------------------------------------------------
# actor model, MLP
# ----------------------------------------------------
# 2 hidden layers, 400 units per layer, tanh output to bound outputs between -1 and 1
class actor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=400):
        super(actor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        #self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        #self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = Variable(Tensor(x))
        #x = F.relu(self.bn1(self.fc1(x)))
        #x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x
        
# ----------------------------------------------------
# critic model, MLP
# ----------------------------------------------------
# 2 hidden layers, 300 units per layer, ouputs rewards therefore unbounded
# Action not to be included until 2nd layer of critic (from paper). Make sure to formulate your critic.forward() accordingly
class critic(nn.Module):
    def __init__(self, state_size, action_size, output_size=1):
        super(critic, self).__init__()
        self.fc1 = nn.Linear(state_size, 300)
        #self.bn1 = nn.BatchNorm1d(300)
        self.fc2 = nn.Linear(300 + action_size, 300)
        self.fc3 = nn.Linear(300, output_size)
        
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3.weight.data.normal_(0, 0.1)

    def forward(self, x, action):
        x = Variable(Tensor(x))
        #x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.fc1(x))
        x = torch.cat([x, action], 1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    
    
class DDPG:
    def __init__(self, input_size=obs_dim, output_size=act_dim, critic_lr=1e-3, actor_lr=1e-4, gamma=GAMMA, batch_size=BATCH_SIZE):
        
        self.gamma = GAMMA
        self.batch_size = BATCH_SIZE
        
        # actor
        self.actor = actor(input_size = obs_dim, output_size = act_dim).type(FloatTensor)
        self.actor_target = actor(input_size = obs_dim, output_size = act_dim).type(FloatTensor)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # critic
        self.critic = critic(state_size = obs_dim, action_size = act_dim, output_size = 1).type(FloatTensor)
        self.critic_target = critic(state_size = obs_dim, action_size = act_dim, output_size = 1).type(FloatTensor)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr = actor_lr)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr = critic_lr, weight_decay=1e-2)
        
        # critic loss
        self.critic_loss = nn.MSELoss()
        
        # noise
        self.noise = OrnsteinUhlenbeckProcess(dimension = act_dim, num_steps = MAX_PATH_LENGTH)

        # replay buffer 
        self.replayBuffer = Replay()
        
        
    def train(self):
     
        # sample from Replay
        s, a, r, next_s, done = self.replayBuffer.sample(self.batch_size)
        
        # update critic (create target for Q function)
        next_q = self.critic_target(next_s, self.actor_target(next_s))
        target_q = Variable(Tensor(r + self.gamma * (1-done) * next_q.data.numpy()))
       
        # critic optimizer and backprop step (feed in target and predicted values to self.critic_loss)
        self.critic.zero_grad()        
        q = self.critic(s, Variable(Tensor(a)))  
        loss = self.critic_loss(q, target_q)
        loss.backward()
        self.optimizer_critic.step()

        # update actor (formulate the loss wrt which actor is updated)
        self.actor.zero_grad()
        loss_actor = -self.critic(s, self.actor(s)).mean()
        
        # actor optimizer and backprop step (loss_actor.backward())
        loss_actor.backward()
        self.optimizer_actor.step()       

        # sychronize target network with fast moving one
        weightSync(self.critic_target, self.critic)
        weightSync(self.actor_target, self.actor)
        
        
        
ddpg = DDPG(input_size = obs_dim, output_size = act_dim)



env = NormalizeAction(env) # remap action values for the environment
avg_val = 0

min_epsilon = 0.01
decay_rate = 5/NUM_EPISODES

#for plotting
running_rewards_ddpg = []
step_list_ddpg = []
step_counter = 0

# set term_condition for early stopping according to environment being used
# term_condition = -150 # Pendulum
# term_condition = 1500 # HalfCheetah
term_condition = 480 # InvertedPendulum

#done = False
#while not done:
#    _, _, done, _ = env.step(np.zeros(act_dim, dtype=np.int))

for itr in range(NUM_EPISODES):
    s = env.reset() # get initial state
    ddpg.noise.reset()
    animate_this_episode = (itr % animate_interval == 0) and VISUALIZE    
    episode_reward = 0
    epsilon = min_epsilon + (1.0 - min_epsilon)*np.exp(-decay_rate*itr)

    while True:
        # use actor to get action, add ddpg.noise.step() to action        
        ddpg.actor.eval()
        action = np.clip(ddpg.actor(np.array([s])).data.numpy() + epsilon * ddpg.noise.step(), -1, 1)
        ddpg.actor.train()
                
        # step action, get next state, reward, done (keep track of total_reward)
        next_s, r, done, _ = env.step(action)
        next_s = next_s.flatten()
        episode_reward += r

        # populate ddpg.replayBuffer
        ddpg.replayBuffer.push([s, action, r, next_s, done])
        s = next_s
        
        ddpg.train()
        step_counter += 1
        
        if done:
            break

    if itr > 100 and avg_val > term_condition:
        break
    
    running_rewards_ddpg.append(episode_reward)
    step_list_ddpg.append(step_counter)

    avg_val = 0.95*avg_val + 0.05*running_rewards_ddpg[-1]
    print("Average value: {} for episode: {}".format(avg_val, itr))
    
    
    
def numpy_ewma_vectorized_v2(data, window):

    alpha = 2/(window + 1.0)
    alpha_rev = 1-alpha
    n = data.shape[0]

    pows = alpha_rev**(np.arange(n+1))
    scale_arr = 1/pows[:-1]
    offset = data[0]*pows[1:]
    pw0 = alpha*alpha_rev**(n-1)

    mult = data.transpose()*pw0*scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums*scale_arr[::-1]
    return out



plt.figure()
out = numpy_ewma_vectorized_v2(np.asarray(running_rewards_ddpg),20)
step_list_ddpg = np.asarray(step_list_ddpg)

plt.plot(step_list_ddpg, out)
plt.title('Training reward for Pendulum-v0 over multiple runs (DDPG)')
plt.xlabel('Number of steps')
plt.ylabel('Cumulative reward')
plt.savefig('DDPG_Pendulum.png')
plt.show()



np.save('DDPG_Pendulum_rewards.npy', np.asarray(running_rewards_ddpg))
np.save('DDPG_Pendulum_steps.npy', np.asarray(step_list_ddpg))