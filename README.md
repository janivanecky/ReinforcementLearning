# ReinforcementLearning

This repo contains basic algorithms/agents used for reinforcement learning. More specifically, you can find here:

- MC control
- Q-learning
- SARSA
- Cross Entropy Method

## Tests

I tested agents on OpenAI gym, CartPole-v0 environment, measuring how long does it take to solve environment (average reward of at least 195
for 100 consecutive episodes). Maximum number of episodes was 1000 and each learning procedure was run 100 times.

| Algorithm  | No. trials without solving | Mean no. episodes to solve | Median no. episodes to solve | Minimum no. episodes to solve |
| ------ | ------: | ------: | ------: | ------: |
| MC         | 28                         | 414                       | 160                          | 91                            |
| Q-learning | 0                          | 394                       | 389                          | 42                            |
| SARSA      | 1                          | 403                       | 348                          | 47                            |
| CEM | 25 | 271 | 11 | 1 |
