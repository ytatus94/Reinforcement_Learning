import gym
import numpy as np

env = gym.make("FronzeLake-v0")

# 環境中的所有狀態
print(env.observation_space.n)
# 環境中執行的動作
print(env.action_space.n)

# 用來找出最佳價值函數
def value_iteration(env, gamma = 1.0):
    # 初始化隨機價值表，先把初始值設為 0
    value_table = np.zeros(env.observation_space.n)
    # 指定遞迴的次數
    no_of_iterations = 100000
    # 設定一個很小的數，用來判斷價值表更新前後的差異
    threshold = 1e-20

    for i in range(no_of_iterations):
        # 每次遞迴的時候把價值表複製到新的價值表上
        updated_value_table = np.copy(value_table)

        for state in range(env.observation_space.n):
            # 建立一個 List 保存在當前狀態下，每個動作對應的 Q 值
            # 每一個元素對應到一個動作
            Q_value = []

            for action in range(env.action_space.n):
                next_states_rewards = []

                for next_sr in env.P[state][action]:
                    # 得到在當前狀態下執行動作後的轉移機率，下一個狀態，獎勵機率
                    trans_prob, next_state, reward_prob, _ = next_sr
                    next_states_rewards.append( (trans_prob * (reward_prob + gamma * updated_value_table[next_state])) )
                Q_value.append( np.sum(next_states_rewards) )
        
            # 找到 Q 的最大值，用來更新價值表
            value_table[state] = max(Q_value)

        # 如果更新前後的價值表之間的差異很小，就停止迭代
        if (np.sum( np.fabs(updated_value_table - value_table) ) <= threshold):
            print("Value-iteration converged at iteration# %d." %(i + 1))
            break

    return value_table

# 用來從最佳的價值函數中抽出策略
def extract_policy(value_table, gamma = 1.0):
    # 初始化隨機策略
    policy = np.zeros(env.observation_space.n)

    for state in range(env.observation_space.n):
        # 把 Q 值表初始化為 0，後面會填入真正的值
        Q_table = np.zeros(env.action_space.n)

        for action in range(env.action_space.n):
            for next_sr in env.P[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                # 計算在這個狀態下，執行特定動作的 Q 值
                Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))

        # 選擇該狀態下，價值最高的動作
        # 最佳策略就是 Q 值最大的動作
        policy[state] = np.argmax(Q_table)

    return policy

optimal_value_function = value_iteration(env=env, gamma=1.0)
optimal_policy = extract_policy(optimal_value_function, gamma=1.0)

print(optimal_policy)
