import gym
# from matplotlib import pyplot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
from functools import partial

# % matplotlib inline
plt.style.use("ggplot")

# 建立 blackjack 環境
env = gym.make("Blackjack-v0")

print(env.action_space)
print(env.observation_space)

# 定義 policy function
# 當點數大於 20 就停牌 (0)
# 當點數小於 20 就要牌 (1)
def sample_policy(observation):
    score, dealer_score, usable_ace = observation
    return 0 if score >= 20 else 1

# 產生 eposide
def generate_episode(policy, env):
    states, actions, rewards = [], [], []
    # 初始化環境，並存於 observation
    observation = env.rest()

    while True:
        states.append(observation)
        action = sample_policy(observation)
        actions.append(action)
        observation, reward, done, info = env.step(action)
        rewards.append(reward)

        # 如果達到最終狀態了，就結束
        if done:
            break
      
    return states, actions, rewards

# 首次訪問 monte carlo 法
# 用來得到各個狀態的價值
def first_visit_mc_prediction(policy, env, n_episodes):
    value_table = defaultdict(float) # 儲存各個狀態的價值
    N = defaultdict(int) # 記錄每個狀態被拜訪的次數

    for _ in range(n_episodes):
        # 得到每個 episode 的狀態與價值
        states, _, rewards = generate_episode(policy, env)
        returns = 0

        for t in range(len(states) - 1, -1, -1):
            R = rewards[t]
            S = states[t]
            returns += R

            if S not in states[:t]:
                N[S] += 1
                value_table[S] += (returns - V[S]) / N[S]

    return value_table

value = first_visit_mc_prediction(sample_policy, env, n_episodes=500000)
print(value)

def plot_blackjack(V, ax1, ax2):
    player_sum = numpy.arange(12, 21 + 1)
    dealer_show = numpy.arange(1, 10 + 1)
    usable_ace = numpy.array([False, True])

    state_values = numpy.zeros((len(player_sum),
                                len(dealer_show),
                                len(usable_ace)))

    for i, player in enumerate(player_sum):
        for j, dealer in enumerate(dealer_show):
            for k, ace in enumerate(usable_ace):
                state_value[i, j, k] = V[player, dealer, ace]

    X, Y = numpy.meshgrid(player_sum, dealer_show)

    ax1.plot_wireframe(X, Y, state_values[:, :, 0])
    ax2.plot_wireframe(X, Y, state_values[:, :, 1])

    for ax in ax1, ax2:
        ax.set_zlim(-1, 1)
        ax.set_ylable("player sum")
        ax.set_xlabel("dealer showing")
        ax.set_zlabel("state-value")

fig, axes = pyplot.subplots(nrows=2, figsize=(5, 8), subplot_kw={"projection": "3d"})
axes[0].set_title("value function without usable ace")
axes[1].set_title("value function with usable ace")
plot_blackjack(value, axes[0], axes[1])
