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



def learnModel():
    T = np.zeros([nS, nA, nS])
    R = np.zeros([nS, nA, nS])
    
    for i in range(100000):
        state = env.reset()
        env.s = np.random.choice(nS)
        
        s = env.s
        a = np.random.choice(nA)
        next_s, reward, _, _ = env.step(a)

        T[s, a, next_s] += 1
        R[s, a, next_s] += reward
    
    for s in range(nS):
        for a in range(nA):
            for next_s in range(nS):
                if T[s, a, next_s] > 0:
                    R[s, a, next_s] /= T[s, a, next_s]
                
            T[s,a,:] /= np.sum(T[s,a,:])

    return R, T

  
[R, T] = learnModel()



def runPolicyEvaluation(policy, V, R, T, discount_factor):
    threshold = 1e-6
    max_iter = 1000
    
    for i in range(max_iter):
        V_new = np.zeros(nS)
        
        for s in range(nS):
            for next_s in range(nS):
                a = policy[s]
                V_new[s] += T[s,a,next_s] * (R[s,a,next_s] + discount_factor * V[next_s])
            
        if np.max(np.abs(V_new - V)) < threshold:
            break
            
        V = V_new

    return V_new



max_iter = 40
discount_factor = 0.98
percentSuccesses = np.zeros(max_iter)

V = np.zeros(nS)
old_policy = np.zeros(nS, dtype=np.int32)


for i in range(max_iter):    
    V_new = runPolicyEvaluation(old_policy, V, R, T, discount_factor)
    policy = np.zeros(nS, dtype=np.int32)
    
    for s in range(nS):       
        action_values = np.zeros(nA)
        
        for a in range(nA):
            for p, next_s, reward, done in env.P[s][a]:                
                action_values[a] += T[s,a,next_s] * (R[s,a,next_s] + discount_factor * V_new[next_s])
                    
        policy[s] = np.argmax(action_values)
    
    V = V_new
    old_policy = policy   
    percentSuccesses[i] = testPolicy(policy)    


    
# plot improvement over time
plt.figure()
plt.bar(np.arange(len(percentSuccesses)), 100*np.array(percentSuccesses), align='center', alpha=0.5)
plt.ylabel('% Episodes Successful')
plt.xlabel('Iteration') 
plt.title('Policy Iteration',fontsize=16)  