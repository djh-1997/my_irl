import numpy as np
import matplotlib.pyplot as plt
import irl.mdp.gridworld as gridworld
from irl.value_iteration import find_policy
import irl.networks as networks

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# 预处理轨迹
def pre_treated(n_states, n_actions, trajectories):
    new_trajectories = []
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
        while visit_max_s in visit_s:  # 退回到最常访问的前一步
            new_trajectory.pop()
            visit_s.pop()
        if len(new_trajectory) > 1:
            new_trajectory.pop()  # 多退一步
        new_trajectories.append(new_trajectory)

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
    return new_trajectories


def supervised_learning(trajectories, policy):
    data = []  # 获取训练数据 [(s,a).....]
    dict = {}  # 记录各状态动作对的访问次数
    for i in trajectories:
        for j in i:
            if j not in data:
                dict[j] = 1
                data.append(j)
            # else:
            #     if dict[j] / len(data) < 0.1:  # 防止某一状态动作对占的比例过大
            #         dict[j] += 1
            #         data.append(j)
    layers = [2, 100, 100, 5]
    grid_size = np.sqrt(len(policy))
    inputs = [np.array([i[0]%grid_size, i[0]//grid_size]).reshape(2,1) for i in data]
    outputs = [np.eye(1, 5, i[1]).reshape(5, 1) for i in data]
    trainData = [(inputs[i], outputs[i]) for i in range(len(inputs))]
    mini_batch_num = 10 #
    epochs = 50000
    learn_rate = 0.01
    error_threshold = 0.15  # 神经网络训练误差允许阈值
    net = networks.ANN(layers)
    net.train(trainData, epochs, mini_batch_num, learn_rate, threshold=error_threshold)
    policy_sl = np.zeros_like(policy)
    for i in range(len(policy)):
        i_input = np.array([i % grid_size, i//grid_size]).reshape(2, 1)
        i_output = net.calc(i_input)
        policy_sl[i] = np.argmax(i_output)
    return policy_sl



def mmp():
    r = 1
    return r


def draw_path(map_grid,paths,title = None):
    # plt.figure(figsize=(map_grid,map_grid))  #为了防止x,y轴间隔不一样长，影响最后的表现效果，所以手动设定等长
    plt.xlim(0, map_grid)
    plt.ylim(0, map_grid)
    my_x_ticks = np.arange(0, map_grid, 1)
    my_y_ticks = np.arange(0, map_grid, 1)
    plt.xticks(my_x_ticks)  # 我理解为竖线的位置与间隔
    plt.yticks(my_y_ticks)
    plt.grid(True)  # 开启栅格
    for path in paths:
        path_x = np.array([0.5+i%map_grid for i in path])
        path_y = np.array([0.5+i//map_grid for i in path])
        plt.plot(path_x, path_y, linewidth=2)
    if title:
        plt.title(title)
    plt.show()


def main(grid_size, discount, trust):  # L正则化系数
    wind = 1-trust  # 专家随机动作系数,专家可能会犯错
    n_trajectories = 6
    trajectory_length = 3*grid_size  # 最大轨迹长度
    gw = gridworld.Gridworld(grid_size, wind, discount)
    ground_r = np.array([gw.reward(s) for s in range(gw.n_states)])  # 真实奖赏函数
    # 由强化学习求最优策略让它代表专家策略产生示例轨迹
    policy = find_policy(gw.n_states, gw.n_actions, gw.transition_probability, ground_r, discount)
    trajectories = gw.generate_trajectories(n_trajectories, trajectory_length, policy, True) # [(s,a,r(s'),s'),....]
    # 画轨迹图 预处理前
    paths = []
    for i in trajectories:
        path = [j[0] for j in i]
        paths.append(path)
    draw_path(gw.grid_size,paths,'预处理前专家示例轨迹')
    # 预处理专家轨迹
    new_trajectories = pre_treated(gw.n_states, gw.n_actions, trajectories)
    # 画轨迹图 预处理后
    paths = []
    for i in new_trajectories:
        path = [j[0] for j in i]
        paths.append(path)
    draw_path(gw.grid_size, paths, '预处理后专家示例轨迹')
    # 监督学习
    policy_sl = supervised_learning(new_trajectories, policy)  # 监督学习
    equal = 0
    for i in range(len(policy)):
        if policy_sl[i] == policy[i]:
            equal += 1/len(policy)
    print("监督学习得到的策略正确率{}%".format(100*equal))
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
    return equal


if __name__ == '__main__':
    equal = main(20, 0.9, 1)