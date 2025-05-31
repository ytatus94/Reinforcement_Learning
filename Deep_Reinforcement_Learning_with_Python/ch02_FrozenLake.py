import gym

env = gym.make('FrozenLake-v0')
env.render()

print(f'env.observation_space={env.observation_space}')
print(f'env.action_space={env.action_space}')

print('Transition probability and reward function:')
print(f'env.P[0][2]={env.P[0][2]}')
print(f'env.P[3][1]={env.P[3][1]}')

state = env.reset()
print(f'env.step(1)={env.step(1)}')
env.render()

random_action = env.action_space.sample()
print(f'random_action={random_action}')

next_state, reward, done, info = env.step(random_action)
print(next_state, reward, done, info)

print('#' * 50)

import gym
env = gym.make('FrozenLake-v0')
state = env.reset()
print('Time Step 0:')
env.render()
num_timesteps = 20
for t in range(num_timesteps):
    random_action = env.action_space.sample()
    next_state, reward, done, info = env.step(random_action)
    print('Time Step {}'.format(t + 1))
    env.render()
    if done:
        break

print('#' * 50)

import gym
env = gym.make('FrozenLake-v0')
num_episodes = 10
num_timesteps = 20
for i in range(num_episodes):
    state = env.reset()
    print('Time Step 0:')
    env.render()
    for t in range(num_timesteps):
        random_action = env.action_space.sample()
        next_state, reward, done, info = env.step(random_action)
        print('Time Step {}'.format(t + 1))
        env.render()
        if done:
            break

