# 必須要先安裝吃角子老虎機的環境
# git clone https://github.com/JKCooper2/gym-bandits.git
# cd gym-bandits
# pip install -e

import gym_bandits
import gym
import numpy as np
import math
import random

# 初始化環境，設定有十隻手臂的 MAB
env = gym.make("BanditTenArmedGaussian-v0")

print(env.action_space)

# 回合次數 (遞迴)
num_rounds = 20000

# 某隻手臂被拉下幾次的計數器
# 有十隻手臂，全部都先初始化為 0
count = np.zeros(10)

# 各手臂的獎勵總和, 每隻手臂分別計算自己的獎勵總和
sum_rewards = np.zeros(10)

# Q 值, 也就是平均獎勵, 這也是每隻手臂分別計算
Q = np.zeros(10)

# Softmax exploration
def softmax(tau):
    total = sum([math.exp(val / tau) for val in Q])
    probs = [math.exp(val / tau) / total for val in Q]
    threshold = random.random()
    cumulative_prob = 0.0

    for i in range(len(probs)):
        cumulative_prob += probs[i]
        if cumulative_prob > threshold:
            return i

    return np.argmax(probs)

# 開始拉手臂
for i in range(num_rounds):
    # 用 epsilon greedy policy 來決定要拉哪一隻手臂
    arm = epsilon_greedy(0.5)
    observation, reward, done, info = env.step(arm)
    # 手臂被拉過就計數一次
    count[arm] += 1
    # 該手臂的獎勵也要累計
    sum_rewards[arm] += reward
    # 計算該手臂的 Q = 該手臂的總獎勵 / 該手臂被拉過的次數
    Q[arm] = sum_rewards[arm] / count[arm]


print("The optimal arm is {}".format(np.argmax(Q)))
