import gym
import pandas as pd
from collections import defaultdict

env = gym.make('Blackjack-v0')

def policy(state):
    return 0 if state[0] > 19 else 1

state = env.reset()
print(f'state={state}')
print(f'state[0]={state[0]}')
print(f'policy(state)={policy(state)}')

num_timesteps = 100

def generate_episode(policy):
    episode = []
    state = env.reset()
    for t in range(num_timesteps):
        action = policy(state)
        next_state, reward, done, info = env.step(action)
        episode.append((state, action, reward))
        if done:
            break
        state = next_state
    return episode

print(f'generate_episode(policy)={generate_episode(policy)}')

# every visit MC prediction
print('Every visit MC prediction')
total_return = defaultdict(float)
N = defaultdict(int)
num_iterations = 500000 # 產生這麼多個 episodes
for i in range(num_iterations):
    episode = generate_episode(policy)
    states, actions, rewards = zip(*episode)
    for t, state in enumerate(states):
        R = (sum(rewards[t:]))
        total_return[state] = total_return[state] + R
        N[state] = N[state] + 1

total_return = pd.DataFrame(
    total_return.items(), 
    columns=['state', 'total_return']
)
N = pd.DataFrame(N.items(), columns=['state', 'N'])

df = pd.merge(total_return, N, on='state')
print(df.head(10))
df['value'] = df['total_return'] / df['N']
print(df.head(10))

print(f'df[df["state"]==(21,9,False)]["value"].values={df[df["state"]==(21,9,False)]["value"].values}')
print(f'df[df["state"]==(5,8,False)]["value"].values={df[df["state"]==(5,8,False)]["value"].values}') 

# first visit MC prediction
print('First visit MC prediction')
total_return = defaultdict(float)
N = defaultdict(int)
num_iterations = 500000
for i in range(num_iterations):
    episode = generate_episode(policy)
    states, actions, rewards = zip(*episode)
    for t, state in enumerate(states):
        if state not in states[0:t]: # first visit MC prediction 要有這一行
            R = (sum(rewards[t:]))
            total_return[state] = total_return[state] + R
            N[state] = N[state] + 1

total_return = pd.DataFrame(
    total_return.items(), 
    columns=['state', 'total_return']
)
N = pd.DataFrame(N.items(), columns=['state', 'N'])

df = pd.merge(total_return, N, on='state')
print(df.head(10))
df['value'] = df['total_return'] / df['N']
print(df.head(10))

print(f'df[df["state"]==(21,9,False)]["value"].values={df[df["state"]==(21,9,False)]["value"].values}')
print(f'df[df["state"]==(5,8,False)]["value"].values={df[df["state"]==(5,8,False)]["value"].values}') 

