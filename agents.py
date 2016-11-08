import numpy as np
import numpy.linalg
import itertools
from random import shuffle
import scipy.signal

def discount(x, gamma): 
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1])[::-1]

class Agent:
    '''
        Parent class for all agents, since they share storing experience and policy 'evaluation'.
        Since all agents assign Q values to percise states, I cannot use continuous states and 
        need to discretize them. Function create_bins() and discretize() take care of that.
        
    '''

    def create_bins(self, min, max, count):
        if min < -10.0:
            min = -10.0
        if max > 10.0:
            max = 10.0
        bins = np.arange(min, max, (max - min) / count)
        return bins

    def discretize(self, values, bins):
        values = values.flatten()
        discretized = []
        for i, v in enumerate(values):
            index = np.digitize(v, bins[i])
            discretized.append(bins[i][np.maximum(index - 1, 0)])
        return np.array(discretized)


    '''
        init_value controls exploration behavior of the agent - if initial estimates 
        of the Q value are overly positive, agent will prefer action it never tried.
    '''
    def init(self, env, init_value = 0):
        self.env = env
        self.observation_bins = []
        for min, max in zip(env.observation_space.low, env.observation_space.high):
            self.observation_bins.append(self.create_bins(min, max, 8))

        self.Q = {}
        for state in list(itertools.product(*self.observation_bins)):
            for a in xrange(env.action_space.n):
                self.Q[tuple(state), a] = init_value
        
        self.sars = []
    
    def experience(self, observation, action, reward, next_observation):
        state = self.discretize(observation, self.observation_bins)
        next_state = self.discretize(next_observation, self.observation_bins)
        self.sars.append([state, action, reward, next_state])

    def get_action(self, observation):
        state = self.discretize(observation, self.observation_bins)
        q_vals = [self.Q[tuple(state), a] for a in xrange(self.env.action_space.n)]
        max_actions = np.argwhere(q_vals == np.amax(q_vals))
        max_actions = np.squeeze(max_actions, 1)
        return max_actions[np.random.randint(max_actions.shape[0])]

class QAgent(Agent):
    '''
        Agent implementing Q-Learning algorithm.
    '''
    def __init__(self, env, init_value = 0):
        self.init(env, init_value)    

    def learn(self, l_rate):
        for [s,a,r,ns] in self.sars:
            maxQ = np.array([self.Q[tuple(ns), na] for na in xrange(self.env.action_space.n)]).max()
            self.Q[tuple(s), a] = self.Q[tuple(s), a] + l_rate * (r + maxQ - self.Q[tuple(s), a])
        self.sars = []

class SarsaAgent(Agent):
    '''
        Agent implementing SARSA algorithm.
    '''
    def __init__(self, env, init_value = 0):
        self.init(env, init_value)    

    def learn(self, l_rate):
        for i, [s,a,r,ns] in enumerate(self.sars):
            na = self.sars[i + 1][1] if i < len(self.sars) - 1 else self.get_action(ns)
            self.Q[tuple(s), a] = self.Q[tuple(s), a] + l_rate * (r + self.Q[tuple(ns), na] - self.Q[tuple(s), a])
        self.sars = []

class MCAgent(Agent):
    '''
        Agent implementing MonteCarlo Q value estimation.
    '''
    def __init__(self, env, init_value = 0):
        self.init(env, init_value)

    def learn(self, l_rate):
        rewards = np.array([sar[2] for sar in self.sars])
        cumulative_rewards = discount(rewards, 1.0)
        self.sars = [[s[0], s[1], c, s[3]] for s, c in zip(self.sars, cumulative_rewards)]
        for [s,a,r,_] in self.sars:
            self.Q[tuple(s), a] = self.Q[tuple(s), a] + l_rate * (r - self.Q[tuple(s), a])
        self.sars = []