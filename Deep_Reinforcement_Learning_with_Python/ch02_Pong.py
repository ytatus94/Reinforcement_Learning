import gym
env = gym.make('Pong-v0')
print(f'env.observation_space={env.observation_space}')
print(f'env.action_space={env.action_space}')
print(env.reset())

env2 = gym.make('Pong-ram-v0')
print(f'env2.observation_space={env2.observation_space}')
print(f'env2.action_space={env2.action_space}')
print(env2.reset())

env3 = gym.make('RoadRunner-v0')
print(f'env3.action_space={env3.action_space}')

