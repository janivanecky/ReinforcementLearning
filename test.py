import numpy as np
import gym
from agents import SarsaAgent, QAgent, MCAgent
        
env = gym.make('CartPole-v0')

initial_value = 300.0
#solver = SarsaAgent(env, initial_value)
#solver = QAgent(env, initial_value)
solver = MCAgent(env, initial_value)

for i in xrange(1000):
    done = False
    total_reward = 0
    observation = env.reset()
    while not done:
        
        action = solver.get_action(observation)
        next_observation, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            reward = -initial_value
        
        solver.experience(observation, action, reward, next_observation)
        observation = next_observation
        
        if done:
            solver.learn(0.1)     

            print 'Iteration: {}; Reward: {}'.format(i, total_reward)
            break