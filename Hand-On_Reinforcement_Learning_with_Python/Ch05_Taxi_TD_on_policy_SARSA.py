import random
import gym

# 建立環境
env = gym.make("Taxi-v1")

# 初始化變數
alpha = 0.85 # learning rate
gamma = 0.90 # discount factor
epsilon = 0.8 # probability for epsilon greedy

# 初始化 Q table
Q = {}
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        Q[(s, a)] = 0.0

# epsilon greedy
def epsilon_greedy(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        # 根據 Q(s, a) 來選取最大的 action
        return max(list(range(action_space.n)), key=lambda x: Q[(state, x)])

for i in range(4000):
    # 總獎勵
    r = 0
    # 每個 episode 都要重置初始狀態
    state = env.reset()
    # 用 epsilon greedy 選擇動作
    action = epsilon_greedy(state, epsilon)

    while True:
        next_state, reward, done, _ = env.step(action)
        # 用 epsilon greedy 選擇下一個動作
        next_action = epsilon_greedy(next_state, epsilon)
        # 更新 Q 值
        Q[(state, action)] += alpha * (reward + gamma * Q[(next_state, next_action)] - Q[(state, action)])
        # 用下一個狀態和下一個動作來更新當前狀態和當前動作
        action = next_action
        state = next_state
        # 儲存所有獎勵
        r += reward

        # 如果已經達到最終狀態就結束
        if done:
            break

env.close()
