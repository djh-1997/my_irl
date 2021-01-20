"""
Run maximum entropy inverse reinforcement learning on the gridworld MDP.

Matthew Alger, 2015
matthew.alger@anu.edu.au
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
from irl.value_iteration import find_policy
import irl.networks as networks
import irl.maxent as maxent
import irl.mdp.gridworld as gridworld

def draw_path(map_grid,paths,title = None):
    #plt.figure(figsize=(map_grid,map_grid))  #为了防止x,y轴间隔不一样长，影响最后的表现效果，所以手动设定等长
    plt.xlim(0,map_grid)
    plt.ylim(0,map_grid)
    my_x_ticks = np.arange(0, map_grid, 1)
    my_y_ticks = np.arange(0, map_grid, 1)
    plt.xticks(my_x_ticks)#我理解为竖线的位置与间隔
    plt.yticks(my_y_ticks)
    plt.grid(True)  #开启栅格
    for path in paths:
        path_x = np.array([0.5+i%map_grid for i in path])
        path_y = np.array([0.5+i//map_grid for i in path])
        plt.plot(path_x,path_y,linewidth=2)
    if title:
        plt.title(title)
    plt.show()

# 预处理轨迹
def pre_treated(n_states, n_actions, trajectories):
    new_trajectories = []
    # 获取最大无重复轨迹
    for trajectory in trajectories:
        new_trajectory = []
        visit_s = []  # 已经访问过的s
        f_s = np.zeros(n_states)
        for (s, a, r_, s_) in trajectory:  # 去除中间无效轨迹，获取骨干（有效）轨迹
            while s in visit_s:
                f_s[s] += 1
                visit_s.pop()
                new_trajectory.pop()
            visit_s.append(s)
            new_trajectory.append((s, a, r_, s_))
        visit_max_s = np.argmax(f_s)  # 到达次数最多的状态
        while visit_max_s in visit_s and visit_max_s != visit_s[0]:  # 退回到最常访问的前一步，并防止退为空
            new_trajectory.pop()
            visit_s.pop()
        if len(new_trajectory) > 1:
            new_trajectory.pop()  # 多退一步
        new_trajectories.append(new_trajectory)
    # # 绘制最大无重复轨迹
    # paths = []
    # for i in new_trajectories:
    #     path = [j[0] for j in i]
    #     paths.append(path)
    # draw_path(np.sqrt(n_states), paths, '最大无重复轨迹')
    # 根据MAP由原始轨迹将轨迹补充到跟原始轨迹等长
    f = np.zeros((n_states, n_actions))
    for trajectory in trajectories:
        for (s, a, _, _) in trajectory:
            f[s][a] += 1
    policy = f.argmax(axis=1)   # 在骨干后面一般访问次数较多，可以利用MAP最大后验概率选择状态动作对，去除噪声干扰

    for new_trajectory, trajectory in zip(new_trajectories, trajectories):  # MAP
        while len(new_trajectory) < len(trajectory):
            found = False  # 是否找到下一状态动作对
            for i in trajectory:
                if i[0] == new_trajectory[-1][3] and i[1] == policy[i[0]]:
                    found = True
                    new_trajectory.append(i)
                    break
            if not found:
                new_trajectory = trajectory
        print(new_trajectory[-4:])
    return new_trajectories


def supervised_learning(trajectories, policy):
    data = []  # 获取训练数据 [(s,a).....]
    dict = {}  # 记录各状态动作对的访问次数
    for i in trajectories:
        for j in i:
            if j not in data:
                dict[j] = 1
                data.append(j)
    layers = [2, 100, 100, 5]
    grid_size = np.sqrt(len(policy))
    inputs = [np.array([i[0]%grid_size, i[0]//grid_size]).reshape(2,1) for i in data]
    outputs = [np.eye(1, 5, i[1]).reshape(5, 1) for i in data]
    trainData = [(inputs[i], outputs[i]) for i in range(len(inputs))]
    mini_batch_num = 10 #
    epochs = 20000
    learn_rate = 0.01
    error_threshold = 0.10  # 神经网络训练误差允许阈值
    net = networks.ANN(layers)
    net.train(trainData, epochs, mini_batch_num, learn_rate, threshold=error_threshold)
    policy_sl = np.zeros_like(policy)
    for i in range(len(policy)):
        i_input = np.array([i % grid_size, i//grid_size]).reshape(2, 1)
        i_output = net.calc(i_input)
        policy_sl[i] = np.argmax(i_output)
    return policy_sl
def main(grid_size, discount, n_trajectories, epochs, learning_rate):
    """
    Run maximum entropy inverse reinforcement learning on the gridworld MDP.

    Plots the reward function.

    grid_size: Grid size. int.
    discount: MDP discount factor. float.
    n_trajectories: Number of sampled trajectories. int.
    epochs: Gradient descent iterations. int.
    learning_rate: Gradient descent learning rate. float.
    """

    wind = 0.1  #模拟干扰，噪声，专家出错导致动作非最优的概率
    trajectory_length = 3*grid_size

    gw = gridworld.Gridworld(grid_size, wind, discount)
    ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])
    # 由强化学习求最优策略让它代表专家策略产生示例轨迹
    policy = find_policy(gw.n_states, gw.n_actions, gw.transition_probability, ground_r, discount)
    trajectories = gw.generate_trajectories(n_trajectories,trajectory_length,policy,random_start=True)
    # 画轨迹图 预处理前
    paths = []
    for i in trajectories:
        path = [j[0] for j in i]
        paths.append(path)
    draw_path(gw.grid_size, paths, '预处理前专家示例轨迹')
    # 预处理专家轨迹
    new_trajectories = pre_treated(gw.n_states, gw.n_actions, trajectories)
    # 画轨迹图 预处理后
    paths = []
    for i in new_trajectories:
        path = [j[0] for j in i]
        paths.append(path)
    draw_path(gw.grid_size, paths, '预处理后专家示例轨迹')

    feature_matrix = gw.feature_matrix()
    trajectories = [[(s, a, r) for (s, a, r, _) in trajectory] for trajectory in trajectories]  # maxent irl处理的格式
    r1, R1 = maxent.irl(feature_matrix, gw.n_actions, discount,
                   gw.transition_probability, np.array(trajectories), epochs, learning_rate)
    r1 = r1 / max(r1)
    loss1 = []
    for r in R1:
        r = r/max(r)
        loss = abs(r-ground_r).sum()
        loss1.append(loss)

    new_trajectories = [[(s, a, r) for (s, a, r, _) in trajectory] for trajectory in new_trajectories]  # maxent irl处理的格式
    feature_matrix = gw.feature_matrix()
    r2, R2 = maxent.irl(feature_matrix, gw.n_actions, discount,
        gw.transition_probability, np.array(new_trajectories), epochs, learning_rate)
    r2 = r2/max(r2)
    loss2 = []
    for r in R2:
        r = r / max(r)
        loss = abs(r - ground_r).sum()
        loss2.append(loss)
    # 监督学习
    policy_sl = supervised_learning(new_trajectories, policy)  # 监督学习
    equal = 0
    for i in range(len(policy)):
        if policy_sl[i] == policy[i]:
            equal += 1 / len(policy)
    print("监督学习得到的策略正确率{}%".format(100 * equal))
    # 由监督学习策略生成轨迹
    sl_trajectories = gw.generate_trajectories(n_trajectories, trajectory_length, policy_sl, random_start=True)
    # 预处理监督学习策略轨迹
    new_sl_trajectories = pre_treated(gw.n_states, gw.n_actions, sl_trajectories)
    # 画轨迹图 监督学习策略
    paths = []
    for i in new_sl_trajectories:
        path = [j[0] for j in i]
        paths.append(path)
    draw_path(gw.grid_size, paths, '监督学习策略估计出的专家轨迹')
    new_sl_trajectories = [[(s, a, r) for (s, a, r, _) in trajectory] for trajectory in new_sl_trajectories]
    mix_trajectories = new_trajectories
    for trajectory in new_sl_trajectories:
        for i in new_trajectories:
            if trajectory[-1] == i[-1]:
                mix_trajectories.append(trajectory)
                break
    feature_matrix = gw.feature_matrix()
    r3, R3 = maxent.irl(feature_matrix, gw.n_actions, discount,
                   gw.transition_probability, np.array(mix_trajectories), epochs, learning_rate)
    r3 = r3 / max(r3)
    loss3 = []
    for r in R3:
        r = r / max(r)
        loss = abs(r - ground_r).sum()
        loss3.append(loss)
    # # 2维图
    # plt.subplot(1, 3, 1)
    # plt.pcolor(r1.reshape((grid_size, grid_size)))
    # plt.colorbar()
    # plt.title("未进行预处理恢复的R")
    # plt.subplot(1, 3, 2)
    # plt.pcolor(r2.reshape((grid_size, grid_size)))
    # plt.colorbar()
    # plt.title("进行预处理恢复的R")
    # plt.subplot(1, 3, 3)
    # plt.pcolor(r3.reshape((grid_size, grid_size)))
    # plt.colorbar()
    # plt.title("预处理且监督学习恢复的R")
    # plt.show()

    # 画三维图
    # 绘图设置

    # X和Y的个数要相同
    X = range(gw.grid_size)
    Y = range(gw.grid_size)
    Z1 = r1
    Z2 = r2
    Z3 = r3
    # meshgrid把X和Y变成平方长度，比如原来都是4，经过meshgrid和ravel之后，长度都变成了16，因为网格点是16个
    xx, yy = np.meshgrid(X, Y)  # 网格化坐标
    X, Y = xx.ravel(), yy.ravel()  # 矩阵扁平化
    # # 设置柱子属性
    height = np.zeros_like(Z1)  # 新建全0数组，shape和Z相同，据说是图中底部的位置
    width = depth = 1  # 柱子的长和宽
    # # 颜色数组，长度和Z一致
    c = ['y'] * len(Z1)

    # 开始画图，注意本来的顺序是X, Y, Z, width, depth, height，但是那样会导致不能形成柱子，只有柱子顶端薄片，所以Z和height要互换
    fig = plt.figure()
    ax = fig.gca(projection='3d')  # 三维坐标轴
    ax.bar3d(X, Y, height, width, depth, Z1, color=c, shade=True)  # width, depth, height
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('reward_vale')
    plt.title("未进行预处理恢复的R")
    plt.show()

    # 开始画图，注意本来的顺序是X, Y, Z, width, depth, height，但是那样会导致不能形成柱子，只有柱子顶端薄片，所以Z和height要互换
    fig = plt.figure()
    ax = fig.gca(projection='3d')  # 三维坐标轴
    ax.bar3d(X, Y, height, width, depth, Z2, color=c, shade=True)  # width, depth, height
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('reward_vale')
    plt.title("预处理后恢复的R")
    plt.show()

    # 开始画图，注意本来的顺序是X, Y, Z, width, depth, height，但是那样会导致不能形成柱子，只有柱子顶端薄片，所以Z和height要互换
    fig = plt.figure()
    ax = fig.gca(projection='3d')  # 三维坐标轴
    ax.bar3d(X, Y, height, width, depth, Z3, color=c, shade=True)  # width, depth, height
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('reward_vale')
    plt.title("预处理且监督学习恢复的R")
    plt.show()

    # 画误差图
    plt.plot(range(epochs), loss1, color='r',  label='未加预处理')
    plt.plot(range(epochs), loss2, color='g', label='加了预处理')
    plt.plot(range(epochs), loss3, color='b', label='预处理且监督学习')
    plt.legend(loc=1)  # 标签展示位置，数字代表标签具位置右上
    plt.xlabel('epochs')
    plt.ylabel('Error')
    plt.title('grid_size=10,discount=0.9')
    plt.plot()
    plt.show()


if __name__ == '__main__':
    main(10, 0.8, 4, 200, 0.01)  # (grid_size, discount, n_trajectories, epochs, learning_rate)
