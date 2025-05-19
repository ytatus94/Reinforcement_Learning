import gym
import numpy as np

env = gym.make("FrozenLake-v0")

def compute_value_function(policy, gamma = 1.0):
    value_table = np.zeros(env.nS)
    # 設定一個很小的數，用來判斷價值表更新前後的差異
    threshold = 1e-10

    with True:
        updated_value_table = np.copy(value_table)

        for state in range(env.nS):
            # 取得當前狀態該執行什麼動作
            action = policy[state]
            # 計算該狀態的價值函數
            value_table[state] = sum([trans_prob * (reward_prob + gamma * updated_value_table[next_state]) for trans_prob, next_state, reward_prob, _ in env.P[state][action]])

            if (np.sum((np.fabs(updated_value_table - value_table))) <= threshold):
                break

        return value_table

    def extract_policy(value_table, gamma = 1.0):
        policy = np.zeros(env.observation_space.n)

        for state in range(env.observation_space.n):
            Q_table = np.zeros(env.action_space.n)
            for action in range(env.action_space.n):
                for next_sr in env.P[state][action]:
                    trans_prob, next_state, reward_prob, _ = next_sr
                    Q_table[action] += (trans_prob * (reward_prob + gamma * value_table[next_state]))

            policy[state] = np.argmax(Q_table)

        return policy

    def policy_iteration(env, gamma = 1.0):
        # 初始化隨機策略
        random_policy = np.zeros(env.observation_space.n)
        # 指定遞迴的次數
        no_of_iterations = 200000
        # 把 discount factor 設為 1 (會尋找未來獎勵)
        gamma = 1.0

        for i in range(no_of_iterations):
            # 策略評估 policy evaluation
            new_value_function = compute_value_function(random_policy, gamma)
            # 抽出策略
            new_policy = extract_policy(new_value_function, gamma)

            # 檢查策略是否收斂
            if (np.all(random_policy == new_policy)):
                print("Policy-Iteration converged at step %d." %(i + 1))
                break

            # 策略改良 policy
            random_policy = new_policy

        return new_policy

    print(policy_iteration(env))
