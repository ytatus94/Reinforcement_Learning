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
        * 例如: 看地圖，選擇走最快的路線
    * Model-free:
        * 不知道 model dynamics
        * 例如: 不看地圖，而是嘗試所有路線後選擇最快的一條
     
* Markov
    * Markov property: 未來只與現在有關，與過去無關
    * Markov chain: 是機率模型，根據當下狀態去預測下一個狀態，和過去狀態無關
    * Markov Decision Process (MDP): 所有 RL 都可以用 MDP 來表示
* Discount factor $\gamma$ 
    * $0 (永不學習, 只考慮當下獎勵) \le \gamma \le 1 (持續學習, 尋找未來獎勵)$
    * 通常用 $0.2 \le \gamma \le 0.8$
* Return 是 agent 收到的總獎勵, i.e. return = sum of rewards
    * $R_{t} = \sum_{i=t+1}^{N} r_{i} = r_{t+1} + r_{t+2} + r_{t+3} + \cdots + r_{N}$
    * 當連續行任務的時候，不會有最終狀態，所以要加入 discount factor $\gamma$: $R_t = \sum_{k}^{\infty} \gamma^{k} r_{t+k+1} = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \gamma^3 r_{t+4} + \cdots$
* Dynamic programming
    * 都是藉由計算 $Q(s, a)$ 來更新 value function 或 policy 
    * Value iteration
        * 一直用 $V^{＊}(s) = \max_{a} Q^{＊}(s, a)$ 直到 $V(s)$ 的變動很小為止
        * 初始化隨機 $V(s) \rightarrow$ 計算 $Q(s, a) \rightarrow$ 使用最大的 $Q(s, a)$ 來更新 $V(s) \rightarrow$ 一直到 $V(s)$ 的變化很小為止
        * 用 $V$ 計算所有 action 的 $Q$, 再從所有 $Q$ 中找出最大的值當作下一輪的 $V$, 重複這樣的循環直到 $V$ 不再改變或改變得很小，此時的 $V$ 就是 optimal $V$. 用 optimal $V$ 計算所有 action 的 $Q$, 再從所有 $Q$ 中最大的 $Q$, 此時對應到的 action 就是最佳的 policy
            * value iteration 是一直循環計算 $Q$ 來更新 $V$ 直到找到 optimal $V$, 然後用 optimal $V$ 來找算 $Q$ 抽出 optimal policy
    * Policy iteration
        * Policy evaluation
            * 初始化隨機 $\pi \rightarrow$ 計算 $V(s)$
        * Policy improvement
            * 用 $Q(s, a)$ 抽出 policy $\rightarrow$ 抽出來的 policy 是否和舊的 policy 一樣, 如果一樣就是收斂了, 如果不一樣就用新的 policy 繼續做 policy evaluation
        * 用給定的 policy 與初始的 $V$ 計算新的 $V$, 把新的 $V $當作下一輪初始的 $V$ 再繼續計算 $V$ 直到 $V$ 不再改變或改變得很小，此時就是 optimal $V$, 用 optimal $V$ 計算所有 action 的 $Q$, 找出最大的 $Q$ 對應到的 action, 就是新的 policy. 比較新舊兩個 policy 是否相同，如果不同那就用新的 policy 和最新的 $V$ 來重複下一個循環的計算，直到新舊兩個 policy 相同，此時的 policy 就是最佳 policy
            * policy iteration 是一直循環計算 $V$ 來找出新的 policy (一樣是用 $Q$ 來抽出 policy), 直到 policy 不再改變時就是 optimal policy
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
* MAB (multi-armed bandit)
    * MAB 就是拉霸機, 吃角子老虎機
        * 每一台拉霸機就只有一隻手臂, 所以 K 倍 MAB 其實就是 K 台拉霸機組合在一起
    * Q = 某一隻手臂的獎勵總和 / 該手臂被拉的次數
        * Optimal $Q = Q(a^{*}) = \max Q(a)$
    * 可用的方法
    * Epsilon greedy
    * 選擇每一隻手臂的機率都是均等的
    * Softmax exploration
    * 選擇每一隻手臂的機率是依照 Boltzmann 機率分配來選擇, 所以又叫做 Boltzmann exploration
    * UBC
    * Thompson sampling

----

## Hand-on Reinforcement Learning with Python
* [Book's repos](https://github.com/PacktPublishing/Hands-On-Reinforcement-Learning-with-Python)
## Deep Reinforcement Learning with Python
* [Book's repos](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-with-Python)
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
