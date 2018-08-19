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



# ----------------------------------------------------
# Policy parametrizing model, MLP
# ----------------------------------------------------
# softmax as activation for output if discrete actions, linear for continuous control
# for the continuous case, output_dim=2*act_dim (each act_dim gets a mean and std_dev)
class mlp(nn.Module):
    def __init__(self, discrete, input_size, output_size, hidden_size=16):
        super(mlp, self).__init__()
        self.discrete = discrete
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        if self.discrete:
            self.fc3 = nn.Linear(hidden_size, output_size)
        else:
            self.fc3 = nn.Linear(hidden_size, output_size*2)
            
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = Variable(Tensor(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if self.discrete:
            x = F.softmax(self.fc3(x), dim=0)
        else:
            x = self.fc3(x)
        return x
    
    
    
def sample_action(logit, discrete):
    # logit is the output of the softmax/linear layer
    if discrete:
        m = torch.distributions.Categorical(logit)
    else:
        l = int(logit.data.shape[0] / 2)
        m = torch.distributions.Normal(logit[l:], F.softplus(logit[:l]))
    action = m.sample()
    log_odds = m.log_prob(action)
    return action, log_odds



def update_policy(paths, net):
    num_paths = len(paths)
    rew_cums = []
    log_odds = []
    policy_loss = []
    
    for path in paths:
        # rew_cums should record return at each time step for each path 
        rews = path['reward']
        R = 0
        for r in rews[::-1]:
            R = r + R
            rew_cums.append(R)
        
        # log_odds should record log_odds obtained at each timestep of path
        los = path['log_odds']
        for lo in los[::-1]:
            log_odds.append(lo)
    
    # make log_odds, rew_cums each a vector
    rew_cums = torch.Tensor(np.asarray(rew_cums))    
    rew_cums = (rew_cums - rew_cums.mean()) / (rew_cums.std() + 1e-5) # create baseline
        
    # calculate policy loss and average over paths
    for log_prob, reward in zip(log_odds, rew_cums):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.cat(policy_loss).sum()
    
    # take optimizer step
    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
    
    
    
# Select Environment
# discrete environment:
# env_name = 'CartPole-v0'
# continous environments:
env_name = 'InvertedPendulum-v1'
# env_name = 'HalfCheetah-v1'

# Make the gym environment
env = gym.make(env_name)
visualize = False
animate = visualize

learning_rate = 1e-3

max_path_length = 500

# Set random seeds
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)

# Saving parameters
logdir='./REINFORCE/test/'

env._max_episode_steps = max_path_length
if visualize:
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    env = gym.wrappers.Monitor(env, logdir, force=True, video_callable=lambda episode_id: episode_id%animate_interval==0)

# Is this env continuous, or discrete?
discrete = isinstance(env.action_space, gym.spaces.Discrete)

# Get observation and action space dimensions
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.n if discrete else env.action_space.shape[0]

# Maximum length for episodes
max_path_length = max_path_length or env.spec.max_episode_steps

# Normalize action
env = NormalizeAction(env)

# Make network object (remember to pass in appropriate flags for the type of action space in use)
net = mlp(discrete, obs_dim, act_dim)

# Make optimizer
optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)



n_iter = 1000 
min_timesteps_per_batch = 2000 # sets the batch size for updating network
avg_reward = 0
avg_rewards = []
step_list_reinforce = []
total_steps = 0
episodes = 0

# term_condition = 450 # CartPole
# term_condition = 1500 # HalfCheetah
term_condition = 480 # InvertedPendulum

#done = False
#while not done:
#    _, _, done, _ = env.step(np.zeros(act_dim, dtype=np.int))

for itr in range(n_iter): # loop for number of optimization steps
    paths = []
    steps = 0
    
    while True: # loop to get enough timesteps in this batch --> if episode ends this loop will restart till steps reaches limit
        ob = env.reset()
        obs, acs, rews, log_odds = [], [], [], []
       
        while True: # loop for episode inside batch
            if animate:
                env.render()
                time.sleep(0.05)
            
            # get parametrized policy distribution from net using current state ob
            logit = net(ob)
            
            # sample action and get log-probability (log_odds) from distribution
            net.eval()
            action, log_prob = sample_action(logit, discrete)
            action = np.clip(action.data.numpy(), -1, 1)
            action = action[0] # for CartPole only
            net.train()
            
            # step environment, record reward, next state
            next_ob, reward, done, _ = env.step(action)
            steps += 1
            
            # append to obs, acs, rewards, log_odds
            obs.append(ob)
            acs.append(action)
            rews.append(reward)
            log_odds.append(log_prob)
            
            # if done, restart episode till min_timesteps_per_batch is reached                     
            if done:
                episodes = episodes + 1
                break
                
            ob = next_ob
                
        path = {"observation" : np.array(obs), 
                "reward" : np.array(rews), 
                "action" : np.array(acs),
                "log_odds" : log_odds}
        
        paths.append(path)
        
        if steps > min_timesteps_per_batch:
            break 
        
    update_policy(paths, net)
    
    if itr == 0:
        avg_reward = path['reward'].sum()
    else:
        avg_reward = 0.95 * avg_reward + 0.05 * path['reward'].sum()
    
    if avg_reward > term_condition:
        break
    
    total_steps += steps
    avg_rewards.append(avg_reward)
    step_list_reinforce.append(total_steps)
    if itr % logging_interval == 0:
        print("Average reward: {} for episode: {}".format(avg_reward, episodes))
        
env.close()



plt.plot(avg_rewards)
plt.title('Training reward for CartPole-v0 over multiple runs (REINFORCE)')
plt.xlabel('Iteration')
plt.ylabel('Average reward')
plt.savefig('REINFORCE_CartPole.png')



np.save('REINFORCE_CartPole-_rewards.npy', np.asarray(avg_rewards))
np.save('REINFORCE_CartPole-_steps.npy', np.asarray(step_list_reinforce))