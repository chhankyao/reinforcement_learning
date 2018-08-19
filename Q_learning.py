import gym
import numpy as np
from matplotlib import pyplot as plt
env = gym.make('FrozenLake-v0')
env = env.unwrapped

nA = env.nA
nS = env.nS



def testPolicy(policy):    
    n_episode = 100
    n_success = 0
    
    for i in range(n_episode):
        s = env.reset()
        done = False
        
        while not done:
            action = policy[s]
            s, reward, done, _ = env.step(action)

        if reward > 0:
            n_success += 1
                
    percentSuccess = n_success / n_episode
    
    return percentSuccess



def runQLearning(learning_rate, discount_factor, num_of_episodes, Q0, explore_type='linear'):

    Q = Q0
    Q_saved = []
    
    for i in range(num_of_episodes):         
        s = env.reset()
        done = False
        
        while not done:
            # Log rate exploration
            if explore_type == 'log':
                thres = 1000 / (1000+i+1)
            else:
                thres = 1 - (i+1)/num_of_episodes
                
            if np.random.rand() > thres:
                action = np.argmax(Q[s,:])
            else:
                action = np.random.choice(nA)
            
            next_s, reward, done, _ = env.step(action)
            Q[s, action] += learning_rate * (reward + discount_factor * np.max(Q[next_s,:]) - Q[s,action])
            s = next_s
        
        if i % (num_of_episodes/10) == 0:
            Q_saved.append(Q.copy())
    
    Q_saved.append(Q.copy())
    return Q_saved



learning_rate = 0.02
discount_factor = 0.95
num_of_episodes = 10000

Q0 = 1e-6 * np.random.rand(nS, nA)
Q_saved = runQLearning(learning_rate, discount_factor, num_of_episodes, Q0, 'log')
percentSuccesses = np.zeros(len(Q_saved))

for i in range(len(Q_saved)):
    policy = np.argmax(Q_saved[i], axis=1)   
    percentSuccesses[i] = testPolicy(policy)

plt.figure()
plt.bar(np.arange(len(percentSuccesses)), 100*np.array(percentSuccesses), align='center', alpha=0.5)
plt.ylabel('% Episodes Successful')
plt.xlabel('Iterations')
plt.title('Log Rate Exploitation', fontsize=16)