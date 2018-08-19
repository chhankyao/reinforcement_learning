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



N = 1000
improvementsIndex = []
percentSuccesses = []

Q_best = 1e-6 * np.random.rand(nS, nA)
percentSuccess_best = 0


for i in range(N):
    Q_test = Q_best + np.random.rand(nS, nA)
    policy = np.argmax(Q_test, axis=1)
    percentSuccess = testPolicy(policy)
    
    if percentSuccess > percentSuccess_best:
        Q_best = Q_test
        percentSuccess_best = percentSuccess
        improvementsIndex.append(i)
        percentSuccesses.append(percentSuccess)

plt.figure()
plt.bar(np.arange(len(improvementsIndex)), 100*np.array(percentSuccesses), align='center', alpha=0.5)
plt.xticks(np.arange(len(improvementsIndex)),improvementsIndex) 
plt.ylabel('% Episodes Successful')
plt.xlabel('Iteration')
plt.title('Hill Climbing',fontsize=16)