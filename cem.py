import numpy as np
import gym

env = gym.make('CartPole-v0')

# return parameters with random values from gaussian distirbution with specified mean and standard deviation
# parameters specify affine transformation and are flattened into a vector
def get_random_theta(means, stddev):
    theta = [np.random.normal(m, s, 1) for m,s in zip(means, stddev)]
    return np.reshape(np.array(theta), (-1))

# compute best action given parameters and state
def get_action(theta, state, action_count):
    W = np.reshape(theta[action_count:], (action_count, state.size))
    b = theta[0:action_count]
    a = np.dot(W, state) + b
    return np.argmax(a)

def episode(theta, render=True):
    observations = env.reset()
    total_reward = 0
    for _ in range(1000):
        if render:
            env.render()
        a = get_action(theta, observations, env.action_space.n)
        observations, reward, done, _ = env.step(a)
        total_reward += reward
        if done: 
            break
    return total_reward

# initial mean and standard deviation for the parameters
theta_size = (env.observation_space.shape[0] + 1) * env.action_space.n
theta_mean = np.zeros((theta_size))
theta_stddev = np.ones((theta_size))

for step in range(100):
    # get new population
    thetas = [get_random_theta(theta_mean, theta_stddev) for _ in xrange(10)]
    # evaluate
    rewards = [episode(theta, render=False) for theta in thetas]
    
    # compute weights used to compute next mean and stddev
    total_reward = np.array(rewards).sum()
    weights = np.array(rewards) / total_reward

    # new mean is a weighted average of population parameters
    theta_mean = np.average(np.array(thetas), 0, weights=weights)
    
    # new stddev ("weighted" stddev)
    theta_variance = np.sum([(thetas[i] - theta_mean)**2 * weights[i] for i in xrange(10)], 0)
    theta_stddev = np.sqrt(theta_variance)
    
    # get new "elite"
    reward = episode(theta_mean, render=False)
    print 'Reward at step {}: {}'.format(step, reward)
