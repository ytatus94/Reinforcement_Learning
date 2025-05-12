import gym

env = gym.make('CartPole-v0')
env.reset() # 初始化環境

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())
