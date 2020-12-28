import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import irl.linear_irl as linear_irl
import irl.mdp.gridworld as gridworld
from irl.value_iteration import find_policy
def main(grid_size,discount,L,trust):#L正则化系数
    wind = 1-trust  #专家随机动作系数,
    trajectory_length = 3*grid_size  #最大轨迹长度
    gw = gridworld.Gridworld(grid_size, wind, discount)
    ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])   #真实奖赏函数
    #policy = [gw.optimal_policy_stochastic(s) for s in range(gw.n_states)]   #采用随机（非确定性）策略，效果没那么好
    #policy = [gw.optimal_policy_deterministic(s) for s in range(gw.n_states)] #采用确定性策略，效果好
    # 由强化学习求最优策略
    policy = find_policy(gw.n_states, gw.n_actions, gw.transition_probability, ground_r, discount, )
    rl1,rl2,rl1l2 = linear_irl.irl(gw.n_states, gw.n_actions, gw.transition_probability,
                       policy, gw.discount, 1, L)  #Rmax=1,L1可变
    return ground_r ,rl1,rl2,rl1l2


if __name__ == '__main__':
    grid_size = 10
    discount= 0.9
    trust = 1
    # 改进的计算误差带权重
    weight = np.array([j+grid_size*i for i in range(grid_size) for j in range(grid_size)])
    # L = np.arange(0, 15, 0.1)
    # sumerror_l1 = np.zeros(len(L))
    # sumerror_l2 = np.zeros(len(L))
    # sumerror_l1l2 = np.zeros(len(L))
    # for i in range(len(L)):
    #     (ground_r, rl1,rl2,rl1l2) = main(grid_size, discount, L[i], trust)  #网格大小10*10，γ=0.2
    #     error_l1 = (ground_r - rl1)
    #     error_l2 = (ground_r - rl2)
    #     error_l1l2 = (ground_r - rl1l2)
    #     a = np.sum(abs(error_l1))
    #     sumerror_l1[i] = a
    #     b = np.sum(abs(error_l2))
    #     sumerror_l2[i] = b
    #     c = np.sum(abs(error_l1l2))
    #     sumerror_l1l2[i] = c
    # plt.plot(L,sumerror_l1, color='r',  label='L1')
    # plt.plot(L, sumerror_l2, color='g', label='L2')
    # plt.plot(L, sumerror_l1l2, color='b', label='L1L2')
    # plt.legend(loc=1)  # 标签展示位置，数字代表标签具位置右上
    # plt.xlabel('L')
    # plt.ylabel('Error')
    # plt.title('grid_size=10,discount=0.9')
    # plt.plot()
    # plt.show()

    (ground_r,rl1,rl2,rl1l2) = main(grid_size,discount,0,1)
    # plt.pcolor(ground_r.reshape((grid_size, grid_size)))
    # plt.colorbar()
    # plt.title("groundtruth reward")
    # plt.show()

    plt.pcolor(rl1.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("Recovered reward")
    plt.show()

    plt.pcolor(rl2.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("L2=10 Recovered reward")
    plt.show()
    plt.pcolor(rl1l2.reshape((grid_size, grid_size)))
    plt.colorbar()
    plt.title("L=10 Recovered reward")
    plt.show()



    np.save('g.npy', ground_r)
    np.save('rl1.npy', rl1)
    np.save('rl2.npy', rl2)
    np.save('rl1l2.npy', rl1l2)

