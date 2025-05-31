import gym
import numpy as np

env = gym.make('FrozenLake-v0')

def compute_value_function(policy):
    num_iterations = 1000
    threshold = 1e-20
    gamma = 1.0
    value_table = np.zeros(env.observation_space.n) # 先把 V 初始化為 0
    for i in range(num_iterations):
        updated_value_table = np.copy(value_table)
        for s in range(env.observation_space.n):
            a = policy[s]
            value_table[s] = sum([prob * (r + gamma * updated_value_table[s_])
               for prob, s_, r, _ in env.P[s][a]])
        if (np.sum(np.fabs(updated_value_table - value_table)) <= threshold):
            break
    return value_table

def extract_policy(value_table):
    gamma = 1.0
    policy = np.zeros(env.observation_space.n)
    for s in range(env.observation_space.n):
        Q_values = [sum([prob * (r + gamma * value_table[s_])
            for prob, s_, r, _ in env.P[s][a]])
            for a in range(env.action_space.n)
        ]
        policy[s] = np.argmax(np.array(Q_values))
    return policy

def policy_iteration(env):
    num_iterations = 1000
    policy = np.zeros(env.observation_space.n) # 使用 random policy 所以先初始化為 0
    for i in range(num_iterations):
        value_function = compute_value_function(policy) # 一直循環直到 V 不再改變或是改變很小
        new_policy = extract_policy(value_function) # 由 V 抽出新的 policy
        if (np.all(policy == new_policy)): # 如果新舊 policy 相同, 那就找到最佳的 policy 了
            break
        policy = new_policy
    return policy

optimal_policy = policy_iteration(env)
print(f'optimal_policy={optimal_policy}')

