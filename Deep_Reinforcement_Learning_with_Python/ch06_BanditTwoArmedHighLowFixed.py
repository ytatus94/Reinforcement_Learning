import gym_bandits
import gym

env = gym.make('BanditTwoArmedHighLowFixed-v0')
print(env.action_space.n)

print(env.p_dist)

