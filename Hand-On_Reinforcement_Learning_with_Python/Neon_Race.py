import gym
import universe
import random

env = gym.make("flashgames.NeonRace-v0")
env.configure(remotes=1) # 自動建立 local 端的 docker container
observation_n = env.reset()

# 控制方向
left = [
  ("KeyEvent", "ArrowUp",    True),
  ("KeyEvent", "ArrowLeft",  True),
  ("KeyEvent", "ArrowRight", False)
]

right = [
  ("KeyEvent", "ArrowUp",    False),
  ("KeyEvent", "ArrowLeft",  False),
  ("KeyEvent", "ArrowRight", True)
]

forward = [
  ("KeyEvent", "ArrowUp",    True),
  ("KeyEvent", "ArrowRight", False),
  ("KeyEvent", "ArrowLeft",  False),
  ("KeyEvent", "n",          True)
]

# 控制要不要轉彎
turn = 0
# 把所有的獎勵存下來
rewards = []
# 設定緩衝區大小
buffer_size = 100
# 設定初始動作是 forward
action = forward

while True:
    turn -= 1
    # 一開始設定是直走 (forward) 然後檢查 turn
    # 如果 turn < 0 表示繼續直走不要轉彎
    if turn <= 0:
        action = forward
        turn = 0

    action_n = [action for ob in observation_n]
    observation_n, reward_n, done_n, info = env.step(action_n)
    rewards += [reward_n[0]]
    
    # 如果平均 reward = 0 就是表示小車一直直走然後卡住了
    # 這代表一定要轉彎才行
    if len(rewards) >= buffer_size:
        mean = sum(rewards) / len(rewards)
        if mean == 0:
            turn = 20
            # 用隨機數值決定要朝哪邊轉彎
            # 小於 0.5 就右轉，大於 0.5 就左轉
            if random.random() < 0.5:
                action = right
            else:
                action = left
            rewards = []

    env.render()
