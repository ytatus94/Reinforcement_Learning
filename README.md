# Reinforcement_Learning Courses

## 重要的觀念
* Agent:
    * Single agent
    * Multiple agent 
* Environment:
    * 環境能用多種方式分類

    |是否能從當下狀態知道結果|是否能知道整個系統的狀態|$s \rightarrow s'$ 的動作數目|動作彼此的相關性|
    |:---|:---|:---|:---|
    |Deterministic environment: 可以根據當下狀態知道結果|Observable environment: 隨時隨地都可以知道整個系統的狀態|Discrete environment: $s \rightarrow s'$ 的動作數目有限|Eposidic (non-sequential) environment: 當下動作不會影響到未來，動作彼此獨立|
    |Stochastic environment: 無法根據當下狀態知道結果|Partial observable environment: 只能知道部分系統的狀態|Continuous environment: $s \rightarrow s'$ 的動作數目無限|Non-eposidic (sequential) environment: 當下動作會影響到未來，動作彼此相關|
  
* Policy $\pi$:
    * 定義了 agent 在環境中的行為規範
    * Deterministic: $\pi(s) = a$ 給定狀態就會執行固定動作
    * Stochastic: $\pi(a|s) = 某機率$ 給定狀態後有一定的機率去執行某動作
* Value function $V(s)$:
    * 又叫做 state-value function, 用來判斷 agent 在某個狀態 $s$ 有多好
    * 等於 agent 從 $s$ 狀態開始到結束收到的總獎勵的期望值
        * $V(s) = E_{\pi}[R_{t}|s_{t}=s]$
        * $V^{\pi}(s) = \sum_{a} \pi(s, a) \sum_{s'} P_{ss'}^{a} [R_{ss'}^{a} + \gamma V^{\pi}(s')]$
    * 可以有很多 value functions, 而最佳的 value function $V^{*}(s)$ 就是價值最高的那一個
        * 最佳的value function 對應到的 policy 就是最佳的 policy
        * $V^{*}(s) = \max_{\pi} V^{\pi}(s)$
        * $V^{＊}(s) = \max_{a} Q^{＊}(s, a) = \max_{a} \sum_{s'} P_{ss'}^{a} [R_{ss'}^{a} + \gamma \sum_{a'} Q^{\pi}(s', a')]$ $\leftarrow$ 這是 Bellman equation for value function
* $Q$ function:
    * 又叫做 station-action value function, 用來判斷 agent 依照策略 $\pi$ 在某狀態 $s$ 執行動做 $a$ 的良好程度
    * $Q^{\pi}(s, a) = E_{\pi}[R_{t}|s_{t}=s, a_{t}=a]$
    * $Q^{\pi}(s, a) = \sum_{s'} P_{ss'}^{a} [R_{ss'}^{a} + \gamma \sum_{a'} Q^{\pi}(s', a')]$ $\leftarrow$ 這是 Bellman equation for $Q$ function
* Model:
    * Model-based:
        * 需要知道 model dynamics, i.e. transition probability
        * 轉移機率 transition probability $P_{ss'}^{a}$
        * 獎勵機率 reward probability $R_{ss'}^{a}$
    * Model-free:
        * 不知道 model dynamics
     
* Markov property: 未來只與現在有關，與過去無關
* Markov Decision Process (MDP): 所有 RL 都是 MDP
* Discount factor $\gamma$ 
    * $0 (永不學習, 只考慮當下獎勵) \le \gamma \le 1 (持續學習, 尋找未來獎勵)$
    * 通常用 $0.2 \le \gamma \le 0.8$
* Return 是 agent 收到的總獎勵, i.e. return = sum of rewards
    * $R_{t} =  = \sum_{i=t+1}^{N} r_{i} = r_{t+1} + r_{t+2} + r_{t+3} + \cdots + r_{N}$
    * 當連續行任務的時候，不會有最終狀態，所以要加入 discount factor $\gamma$: $R_t = \sum_{k}^{\infty} \gamma^{k} r_{t+k+1} = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \gamma^3 r_{t+4} + \cdots$
* Dynamic programming
    * 都是藉由計算 $Q(s, a)$ 來更新 value function 或 policy 
    * Value iteration: 一直用 $V^{＊}(s) = \max_{a} Q^{＊}(s, a)$ 直到 $V(s)$ 的變動很小為止
    * Policy iteration
        * Policy evaluation
        * Policy improvement: 用 $Q(s, a)$ 抽出 policy
* Monte Carlo
    * Monte Carlo prediction
        * First visit: 只有第一次訪問該狀態的時候才要計算平均回報
        * Every visit: 每次訪問到該狀態的時候都要計算平均回報
    * Monte Carlo contral
        * On policy: $\epsilon$ greedy
            * 用 exploration-exploitation, 當機率比 $\epsilon$ 小的時候就用隨機動作 (exploration), 當機率比 $\epsilon$ 大時就用最佳動作 (exploitation)
        * Off policy: 用兩個 policies: bahavior policy $\mu$ 和 target policy $\pi$
            * $\mu$ 和 $\pi$ 之間有一個比例關係, 利用 important sampling 來做加權 
* Temporal difference (TD)
    * TD prediction
        * $V(s) = V(s) + \alpha [r + \gamma V(s') - V(s)]$
            * 實際獎勵 $r + \gamma V(s')$
            * 期望獎勵 $V(s)$
            * TD error = $r + \gamma V(s') - V(s)$
    * TD contral
        * On policy: $Q$-learning
            * $Q(s,a) = Q(s, a) + \alpha [r + \gamma \max Q(s', a') - Q(s, a)]$
            * 動作的選擇是用 $\epsilon$ greedy
            * 更新 $Q$ 值的時候用最大值 $\max Q(s', a')$
        * SARSA:
            * $Q(s,a) = Q(s, a) + \alpha [r + \gamma Q(s', a') - Q(s, a)]$
            * 動作的選擇是用 $\epsilon$ greedy
            * 更新 $Q$ 值的時候用 $\epsilon$ greedy $Q(s', a')$

----

## Hand-on Reinforcement Learning with Python
* [Book's repos](https://github.com/PacktPublishing/Hands-On-Reinforcement-Learning-with-Python)
## Deep Reinforcement Learning with Python
* Book's repos:
## Deep Reinforcement Learning Hands-On
* Book's repos:
## Artificial Intelligence: Reinforcement Learning in Python
* Udemy course by Lazy Progammer
## Cutting Edge AI: Deep Reinforcement Learning in Python
* Udemy course by Lazy Progammer
## Advanced AI: Deep Reinforcement Learning in Python
* Udemy course by Lazy Progammer
## Deep Reinforcement Learning: Hands-on AI tutorial in Python
## Morden Reinforcement Learning: Actor-Critic Agents
