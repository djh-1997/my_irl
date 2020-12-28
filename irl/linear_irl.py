import random
import numpy as np
from cvxopt import matrix, solvers
def irl(n_states, n_actions, transition_probability, policy, discount,Rmax,L):
    A = set(range(n_actions))  # 创建一个无序不重复动作集合
    transition_probability = np.transpose(transition_probability, (1, 0, 2))
    Pa_star =np.array([transition_probability[policy[s], s]for s in range(n_states)])#Pa*
    def T(a, s):  # 1*N
        # return np.dot(transition_probability[policy[s], s] - transition_probability[a, s],
        #               np.linalg.inv(np.eye(n_states) - discount * Pa_star))  # r(s) 计算(Pa*|s-Pa|s)（I-γPa*)^-1

        return np.dot(transition_probability[policy[s], s] -transition_probability[a, s],
                      np.dot(np.linalg.inv(np.eye(n_states) -discount * Pa_star),Pa_star))  #r(s') 计算(Pa*|s-Pa|s)（I-γPa*)^-1*Pa*
    # Minimize c . x.
    # L1正则化求
    # -I R <= Rmax 1
    # I R <= Rmax 1
    c = -np.hstack([np.zeros(n_states), np.ones(n_states), -L * np.ones(n_states)])  # 垂直堆叠
    G = np.vstack([
        np.hstack([np.vstack([-T(a, s) for s in range(n_states) for a in A - {policy[s]}]),
                   np.vstack([np.eye(1, n_states, s) for s in range(n_states) for a in A - {policy[s]}]),
                   np.zeros((n_states * (n_actions - 1), n_states))]),
        np.hstack([np.vstack([-T(a, s) for s in range(n_states) for a in A - {policy[s]}]),
                   np.zeros((n_states * (n_actions - 1), n_states)),
                   np.zeros((n_states * (n_actions - 1), n_states))]),
        np.hstack([-np.eye(n_states), np.zeros((n_states, n_states)), -np.eye(n_states)]),
        np.hstack([np.eye(n_states), np.zeros((n_states, n_states)), -np.eye(n_states)]),
        np.hstack([np.zeros((n_states, n_states)), np.zeros((n_states, n_states)), np.eye(n_states)])
    ])
    h = np.vstack([np.zeros((8 * n_states, 1)), Rmax * np.ones((n_states, 1))])
    results = solvers.lp(matrix(c), matrix(G), matrix(h))  # 凸优化包cvxopt,里的线性规划求解包含R的向量  公式16，17
    rl1 = np.asarray(results["x"][:n_states], dtype=np.double).reshape((n_states,))  # 取出结果中的reward

    #L2正则化求
    P = np.hstack([
        np.vstack([L*np.eye(n_states),np.zeros((n_states,n_states)),np.zeros((n_states,n_states))]),
        np.vstack([np.zeros((n_states, n_states)), np.zeros((n_states, n_states)), np.zeros((n_states, n_states))]),
        np.vstack([np.zeros((n_states, n_states)), np.zeros((n_states, n_states)), np.zeros((n_states, n_states))])
    ])
    q = -np.hstack([np.zeros(n_states), np.ones(n_states),np.zeros(n_states)])  #垂直堆叠
    results = solvers.qp(matrix(P), matrix(q), matrix(G),matrix(h))
    rl2 = np.asarray(results["x"][:n_states], dtype=np.double).reshape((n_states,))

    #L1 L2正则化求
    q = -np.hstack([np.zeros(n_states), np.ones(n_states),
                    -L*np.ones(n_states)])  # 垂直堆叠
    results = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h))

    rl1l2 = np.asarray(results["x"][:n_states], dtype=np.double).reshape((n_states,))

    return rl1,rl2,rl1l2

def large_irl(values, transition_probability, feature_matrix, n_states,
              n_actions, policy, L):
    D = feature_matrix.shape[1]  # 取feature_matrix第二维数D，即状态的维数
    v = np.zeros((n_states, n_actions - 1, D))
    for i in range(n_states):
        a1 = policy[i]
        exp_on_policy = np.dot(transition_probability[i, a1],values)  # 1*D 状态i最优动作下各状态基函数的值函数
        seen_policy_action = False
        for j in range(n_actions):
            # Skip this if it's the on-policy action.
            if a1 == j:
                seen_policy_action = True
                continue
            exp_off_policy = np.dot(transition_probability[i, j],values)
            if seen_policy_action:
                v[i, j - 1] = exp_on_policy - exp_off_policy
            else:
                v[i, j] = exp_on_policy - exp_off_policy

    # L1正则化求
    q = np.hstack([-np.ones(n_states), np.zeros((n_actions - 1) * n_states * 2),L*np.ones(D),np.zeros(D)]) #L为L1正则化系数
    x_size = n_states + (n_actions - 1) * n_states * 2 + 2*D
    assert q.shape[0] == x_size

    A = np.hstack([
        np.zeros((n_states * (n_actions - 1), n_states)),
        np.eye(n_states * (n_actions - 1)),
        -np.eye(n_states * (n_actions - 1)),
        np.zeros((n_states * (n_actions - 1), D)),
        -np.vstack([v[i, j].T for i in range(n_states)
                   for j in range(n_actions - 1)])])
    assert A.shape[1] == x_size
    b = np.zeros(A.shape[0])
    G = np.vstack([
        np.hstack([
            np.zeros((D, n_states)),
            np.zeros((D, n_states * (n_actions - 1))),
            np.zeros((D, n_states * (n_actions - 1))),
            np.zeros((D, D)),
            np.eye(D)]),
        np.hstack([
            np.zeros((D, n_states)),
            np.zeros((D, n_states * (n_actions - 1))),
            np.zeros((D, n_states * (n_actions - 1))),
            np.zeros((D, D)),
            -np.eye(D)]),
        np.hstack([
            np.zeros((D, n_states)),
            np.zeros((D, n_states * (n_actions - 1))),
            np.zeros((D, n_states * (n_actions - 1))),
            -np.eye(D),
            np.eye(D)]),
        np.hstack([
            np.zeros((D, n_states)),
            np.zeros((D, n_states * (n_actions - 1))),
            np.zeros((D, n_states * (n_actions - 1))),
            -np.eye(D),
            -np.eye(D)]),
        np.hstack([
            np.zeros((n_states * (n_actions - 1), n_states)),
            -np.eye(n_states * (n_actions - 1)),
            np.zeros((n_states * (n_actions - 1), n_states * (n_actions - 1))),
            np.zeros((n_states * (n_actions - 1), D)),
            np.zeros((n_states * (n_actions - 1), D))]),
        np.hstack([
            np.zeros((n_states * (n_actions - 1), n_states)),
            np.zeros((n_states * (n_actions - 1), n_states * (n_actions - 1))),
            -np.eye(n_states * (n_actions - 1)),
            np.zeros((n_states * (n_actions - 1), D)),
            np.zeros((n_states * (n_actions - 1), D))]),
        np.hstack([
            np.vstack([np.eye(1, n_states, s) for s in range(n_states)  for i in range(n_actions-1)]),
            -np.eye(n_states*(n_actions-1)),
            2*np.eye(n_states * (n_actions - 1)),
            np.zeros((n_states * (n_actions - 1),D)),
            np.zeros((n_states * (n_actions - 1),D))
        ])
    ])
    assert G.shape[1] == x_size
    h = np.vstack([np.ones((D * 2, 1)),np.zeros((n_states * (n_actions - 1) *3+2*D, 1))])
    P = np.zeros((q.shape[0], q.shape[0]))
    results = solvers.qp(matrix(P),matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))
    alpha1 = np.asarray(results["x"][-D:], dtype=np.double)
    rl1 = np.dot(feature_matrix, alpha1)

    #L1L2正则化求
    P = np.hstack([
        np.zeros((q.shape[0],q.shape[0]-D)),
        np.vstack([np.zeros((q.shape[0]-D, D)), L*np.eye(D)])
    ])
    results = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))
    alpha12 = np.asarray(results["x"][-D:], dtype=np.double)
    rl1l2= np.dot(feature_matrix, alpha12)

    #L2正则化求
    q = np.hstack(
        [-np.ones(n_states), np.zeros((n_actions - 1) * n_states * 2), np.zeros(D), np.zeros(D)])  # L为L1正则化系数
    results = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))
    alpha2 = np.asarray(results["x"][-D:], dtype=np.double)
    rl2 = np.dot(feature_matrix,alpha2)
    return rl1.T, rl2.T, rl1l2.T