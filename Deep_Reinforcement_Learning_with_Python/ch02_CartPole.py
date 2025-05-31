import gym
env = gym.make('CartPole-v0')
#env.render()
print(f'env.reset()={env.reset()}')
print(f'env.observation_space={env.observation_space}')
print(f'env.observation_space.high={env.observation_space.high}')
print(f'env.observation_space.low={env.observation_space.low}')
print(f'env.action_space={env.action_space}')
env.close()

print('*' * 50)

import gym
env = gym.make('CartPole-v0')

num_episodes = 100
num_timesteps = 50

for i in range(num_episodes):
    Return = 0
    state = env.reset()
    for t in range(num_timesteps):
        #env.render()
        random_action = env.action_space.sample()
        next_state, reward, done, info = env.step(random_action)
        Return = Return + reward
        if done:
            break
    if i % 10 == 0:
        print('Episode: {}, Return: {}'.format(i, Return))
env.close()

