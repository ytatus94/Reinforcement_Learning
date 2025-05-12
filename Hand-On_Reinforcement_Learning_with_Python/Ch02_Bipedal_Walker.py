import gym

env = gym.make('BipedalWalker-v2')

for episode in range(100): # 跑 100 個 episodes
    observation = env.reset() # 初始化
    for i in range(10000): # 每個 episode 走 10000 步
        env.render()
        print(observation)
        
        action = env.action_space.sample() # 從 action space 中隨機取樣選一個動作
        observation, reward, done, info = env.step(action)

        if done: # 如果 done 是 True 表示 episode 已經結束了
            print("{} timesteps taken for the Episode".format(i+1))
            break
