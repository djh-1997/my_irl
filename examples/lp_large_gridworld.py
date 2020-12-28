"""
Run large state space linear programming inverse reinforcement learning on the
gridworld MDP.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""
import numpy as np
import matplotlib.pyplot as plt
import irl.linear_irl as linear_irl
import irl.mdp.gridworld as gridworld
from irl.value_iteration import value
from irl.value_iteration import find_policy
def main(grid_size, discount,L):
    wind = 0.3
    trajectory_length = 3*grid_size
    gw = gridworld.Gridworld(grid_size, wind, discount)
    ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])
    #policy = [gw.optimal_policy_deterministic(s) for s in range(gw.n_states)]#由确定性最优策略(自己预先设置的）求R
    #由强化学习求最优策略
    policy = find_policy(gw.n_states, gw.n_actions, gw.transition_probability, ground_r, discount,)
    # Need a value function for each basis function.
    feature_matrix = gw.feature_matrix()
    values = []
    for dim in range(feature_matrix.shape[1]):
        reward = feature_matrix[:, dim]
        values.append(value(policy, gw.n_states, gw.transition_probability,
                            reward, gw.discount))
    values = np.array(values).T

    rl1,rl2,rl1l2 = linear_irl.large_irl(values, gw.transition_probability,
                        feature_matrix, gw.n_states, gw.n_actions, policy,L)
    return ground_r,rl1,rl2,rl1l2


if __name__ == '__main__':
    grid_size = 10
    discount = 0.9
    L = np.arange(0, 40, 4)
    weight = np.array([j + grid_size * i for i in range(grid_size) for j in range(grid_size)])
    sumerror_l1 = np.zeros(len(L))
    sumerror_l2 = np.zeros(len(L))
    sumerror_l1l2 = np.zeros(len(L))
    # for i in range(len(L)):
    #     (ground_r, rl1,rl2,rl1l2) = main(grid_size, discount, L[i])  #网格大小10*10，
    #     error_l1 = (ground_r - rl1) #改进的计算误差带权重
    #     error_l2 = (ground_r - rl2)
    #     error_l1l2 = (ground_r - rl1l2)
    #     a = np.sum(abs(error_l1))
    #     sumerror_l1[i] = a
    #     b = np.sum(abs(error_l2))
    #     sumerror_l2[i] = b
    #     c = np.sum(abs(error_l1l2))
    #     sumerror_l1l2[i] = c
    #
    #
    # plt.plot(L,sumerror_l1, color='r',  label='L1')
    # plt.plot(L, sumerror_l2, color='g', label='L2')
    # plt.plot(L, sumerror_l1l2, color='b', label='L1L2')
    # plt.legend(loc=1)  # 标签展示位置，数字代表标签具位置右上
    # plt.xlabel('L')
    # plt.ylabel('Error')
    # plt.title('grid_size=10,discount=0.9')
    # plt.plot()
    # plt.show()

    (ground_r,rl1,rl2,rl1l2) = main(grid_size,discount,8)
    plt.pcolor(ground_r.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("L1=1Recovered reward")
    plt.show()
    plt.pcolor(rl1.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("L1=1Recovered reward")
    plt.show()
    plt.pcolor(rl2.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("L2=1Recovered reward")
    plt.show()
    plt.pcolor(rl1l2.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("L=1Recovered reward")
    plt.show()
