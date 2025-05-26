import random
import gym

# 建立環境
env = gym.make("Taxi-v1")
# 畫出環境
env.render()

# 初始化變數
alpha = 0.4 # learning rate
gamma = 0.999 # discount factor
epsilon = 0.017 # probability for epsilon greedy

# 初始化 Q table
q = {}
for s in range(env.observation_space.n):
    for a in range(env.action_space.n):
        q[(s, a)] = 0

def update_q_table(prev_state, action, reward, next_state, alpha, gamma):
    qa = max([q[next_state, a] for a in range(env.action_space.n)])
    q[(prev_state, a)] += alpha * (reward + gamma * qa - q[(prev_state, action)])

def epslion_greedy_policy(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        retrun max(list(env.action_space.n), key=lambda x: q[(state, x)])

for i in range(8000):
    r = 0
    prev_state = env.reset()

    while True:
        env.render()
        # 用 epsilon greedy 來選擇動作
        action = epslion_greedy_policy(prev_state, epsilon)
        # 執行動作來得到獎勵
        next_state, reward, done, _ = env.step(action)
        # 更新 Q table
        update_q_table(prev_state, action, reward, next_state, alpah, gamma)
        # 更新狀態
        prev_state = next_state
        # 儲存所有獎勵
        r += reward

        # 如果已經達到最終狀態就結束
        if done:
            break

    print("Total reward: ", r)

env.close()
