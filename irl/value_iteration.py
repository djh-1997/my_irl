import numpy as np
def value(policy, n_states, transition_probabilities, reward, discount,
                    threshold=1e-2):  #计算策略的值函数  r为r(s')不是r(s,a)
    v = np.zeros(n_states)
    diff = float("inf")
    while diff > threshold:
        diff = 0
        for s in range(n_states):
            vs = v[s]
            a = policy[s]
            v[s] = sum(transition_probabilities[s, a, k] *
                       (reward[k] + discount * v[k])
                       for k in range(n_states))  #r为r(s')不是r(s,a)
            diff = max(diff, abs(vs - v[s]))  #最大状态值变化（收敛时变化很小）
    return v

def optimal_value(n_states, n_actions, transition_probabilities, reward,
                  discount, threshold=1e-2):  #获取最优值函数   r为r(s')不是r(s,a)
    v = np.zeros(n_states)
    diff = float("inf")
    while diff > threshold:
        diff = 0
        for s in range(n_states):
            max_v = float("-inf")
            for a in range(n_actions): #选择最优动作的值函数作为最优值函数（类似Q学习）
                tp = transition_probabilities[s, a, :]
                max_v = max(max_v, np.dot(tp, reward + discount*v)) #r为r(s')不是r(s,a)
            new_diff = abs(v[s] - max_v)
            if new_diff > diff:
                diff = new_diff
            v[s] = max_v
    return v

def find_policy(n_states, n_actions, transition_probabilities, reward, discount,threshold=1e-2, v=None, stochastic=False):
    if v is None:
        v = optimal_value(n_states, n_actions, transition_probabilities, reward,
                          discount, threshold)

    if stochastic: #随机策略   9.2 from Ziebart's thesis.
        Q = np.zeros((n_states, n_actions))
        for i in range(n_states):
            for j in range(n_actions):
                p = transition_probabilities[i, j, :]
                Q[i, j] = p.dot(reward + discount*v)  #r为r(s')不是r(s,a)
        Q -= Q.max(axis=1).reshape((n_states, 1))
        Q = np.exp(Q)/np.exp(Q).sum(axis=1).reshape((n_states, 1))
        return Q #概率与Q(s,a)的指数次成正比

    def _policy(s):  #确定策略 求状态s下的最优动作a
        return max(range(n_actions),
                   key=lambda a: sum(transition_probabilities[s, a, k] *
                                     (reward[k] + discount * v[k])
                                     for k in range(n_states)))
    policy = np.array([_policy(s) for s in range(n_states)])
    return policy

if __name__ == '__main__':
    # Quick unit test using gridworld.
    import mdp.gridworld as gridworld
    gw = gridworld.Gridworld(3, 0.3, 0.9)
    v = value([gw.optimal_policy_deterministic(s) for s in range(gw.n_states)],
              gw.n_states,
              gw.transition_probability,
              [gw.reward(s) for s in range(gw.n_states)],
              gw.discount)
    #print(v)
    assert np.isclose(v,
                      [5.7194282, 6.46706692, 6.42589811,
                       6.46706692, 7.47058224, 7.96505174,
                       6.42589811, 7.96505174, 8.19268666], 1).all()
    opt_v = optimal_value(gw.n_states,
                          gw.n_actions,
                          gw.transition_probability,
                          [gw.reward(s) for s in range(gw.n_states)],
                          gw.discount)
    #print(opt_v)
    assert np.isclose(v, opt_v).all()
    #policy = [gw.optimal_policy_deterministic(s) for s in range(gw.n_states)]
    #print(policy)
    #gw.generate_trajectories(6, 9, policy, random_start=False)
