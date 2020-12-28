import numpy as np
import numpy.random as rn
class Gridworld(object):
    def __init__(self, grid_size, wind, discount):
        self.actions = ((1, 0), (0, 1), (-1, 0), (0, -1), (0, 0)) #4个动作，0，x+1，1，y+1，2，x-1，3，y-1，4，不变
        self.n_actions = len(self.actions)
        self.n_states = grid_size**2                        #状态个数
        self.grid_size = grid_size
        self.wind = wind
        self.discount = discount
        self.transition_probability = np.array(
            [[[self._determined_transition_probability(i, j, k)
               for k in range(self.n_states)]
              for j in range(self.n_actions)]
             for i in range(self.n_states)])    #状态转移矩阵
    def __str__(self):
        return "Gridworld({}, {}, {})".format(self.grid_size, self.wind,self.discount)

    def feature_vector(self, i, feature_map="ident"):
        if feature_map == "coord":   #i所在行列+1
            f = np.zeros(self.grid_size)
            x, y = i % self.grid_size, i // self.grid_size
            f[x] += 1
            f[y] += 1
            return f
        if feature_map == "proxi": #其到i的横纵向距离
            f = np.zeros(self.n_states)
            x, y = i % self.grid_size, i // self.grid_size
            for b in range(self.grid_size):
                for a in range(self.grid_size):
                    dist = abs(x - a) + abs(y - b)
                    f[self.point_to_int((a, b))] = dist
            return f
        # Assume identity map.
        f = np.zeros(self.n_states)
        f[i] = 1
        return f

    def feature_matrix(self, feature_map="ident"):
        features = []
        for n in range(self.n_states):
            f = self.feature_vector(n, feature_map)
            features.append(f)
        return np.array(features)

    def int_to_point(self, i):
        return (i % self.grid_size, i // self.grid_size)

    def point_to_int(self, p):
        return p[0] + p[1]*self.grid_size

    def neighbouring(self, i, k):
        return abs(i[0] - k[0]) + abs(i[1] - k[1]) <= 1

    def _transition_probability(self, i, j, k):#非确定性MDP（执行一个动作，可以到几种状态，不是唯一确定的状态）
        """
        感觉这个有问题，gridworld应该就是一个确定性的，给一个动作只能有一个确定的下一状态
        而且在这个环境里取专家轨迹也是这样确定性取的，没有考虑一个动作可能会小几率到随机状态（相邻）
        """
        xi, yi = self.int_to_point(i)
        xj, yj = self.actions[j]
        xk, yk = self.int_to_point(k)

        if not self.neighbouring((xi, yi), (xk, yk)): #i和k不相邻，转移概率为0
            return 0.0
        if (xi + xj, yi + yj) == (xk, yk):  # Is k the intended state to move to?
            return 1 - self.wind + self.wind/self.n_actions

        # If these are not the same point, then we can move there by wind.随机动作
        if (xi, yi) != (xk, yk):
            return self.wind/self.n_actions

        # If these are the same point, we can only move here by either moving
        # off the grid or being blown off the grid. Are we on a corner or not?
        if (xi, yi) in {(0, 0), (self.grid_size-1, self.grid_size-1),
                        (0, self.grid_size-1), (self.grid_size-1, 0)}:
            # Corner.
            # Can move off the edge in two directions.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < self.grid_size and
                    0 <= yi + yj < self.grid_size):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here plus an extra chance of blowing
                # onto the *other* off-grid square.
                return 1 - self.wind + 2*self.wind/self.n_actions
            else:
                # We can blow off the grid in either direction only by wind.
                return 2*self.wind/self.n_actions
        else:
            # Not a corner. Is it an edge?
            if xi not in {0, self.grid_size-1} and yi not in {0, self.grid_size-1}:
                # Not an edge.
                return 0.0

            # Edge.
            # Can only move off the edge in one direction.
            # Did we intend to move off the grid?
            if not (0 <= xi + xj < self.grid_size and
                    0 <= yi + yj < self.grid_size):
                # We intended to move off the grid, so we have the regular
                # success chance of staying here.
                return 1 - self.wind + self.wind/self.n_actions
            else:
                # We can blow off the grid only by wind.
                return self.wind/self.n_actions
    def _determined_transition_probability(self, i, j, k):#确定性MDP（执行一个动作，只到1种状态）
        xi, yi = self.int_to_point(i)
        xj, yj = self.actions[j]
        xk, yk = self.int_to_point(k)
        if (xi + xj, yi + yj) == (xk, yk):  # Is k the intended state to move to?
            return 1
        elif ((xi, yi) == (xk, yk)) and (not(0 <= xi + xj < self.grid_size and
                    0 <= yi + yj < self.grid_size)): #反弹回原状态
            return 1
        else:
            return 0.0

    def reward(self, state_int): #真实奖赏函数
        if state_int == self.n_states //2 + np.sqrt(self.n_states)//2: #网格正中间
            return 1
        # if state_int == self.n_states-1: #网格右上角
        #     return 1

        return 0

    def average_reward(self, n_trajectories, trajectory_length, policy):
        trajectories = self.generate_trajectories(n_trajectories,trajectory_length, policy)
        rewards = [[r for _, _, r in trajectory] for trajectory in trajectories]
        rewards = np.array(rewards)

        # Add up all the rewards to find the total reward.
        total_reward = rewards.sum(axis=1) #按行相加(为什么不是γ*)

        # Return the average reward and standard deviation.
        return total_reward.mean(), total_reward.std()  #返回平均R和标准差

    def optimal_policy_stochastic(self, state_int):   #（非确定性）最优策略为右移或者上移
        sx, sy = self.int_to_point(state_int)
        if sx < self.grid_size-1 and sy < self.grid_size-1: #状态不在边缘，随机选择右移或上移
            return rn.randint(0, 2)  #随机返回动作0或1（分别代表右移或上移）
        if sx == self.grid_size-1 and sy == self.grid_size-1: #状态目标点，随机选择右移或上移
            return rn.randint(0, 2)  #随机返回动作0或1（分别代表右移或上移）
        if sy == self.grid_size-1:
            return 0              #动作0，x坐标+1，即右移一步
        if sx == self.grid_size-1:
            return 1              #动作1，坐标+1，即上移一步
        raise ValueError("Unexpected state.")

    def optimal_policy_deterministic(self, state_int):#确定性最优策略
        sx, sy = self.int_to_point(state_int)
        if sx < sy:
            return 0  #左上角右移
        return 1  #右下角上移
    def generate_trajectories(self, n_trajectories, trajectory_length,policy = None,random_start=False ):#由专家策略生成专家轨迹
        trajectories = []
        for _ in range(n_trajectories):
            if random_start:
                sx, sy = rn.randint(self.grid_size), rn.randint(self.grid_size)
            else:
                sx, sy = 0, 0
            trajectory = []
            for _ in range(trajectory_length):
                if rn.random() < self.wind:
                    #专家策略可能不是最优的，有一定的概率随机移动
                    action = self.actions[rn.randint(0, 5)]
                else:
                    if policy is None:
                        action = self.actions[self.optimal_policy_deterministic((self.point_to_int((sx, sy))))]
                    else: #给了policy optimal_policy_deterministic
                        action = self.actions[policy[self.point_to_int((sx, sy))]]
                if (0 <= sx + action[0] < self.grid_size and 0 <= sy + action[1] < self.grid_size):
                    next_sx = sx + action[0]
                    next_sy = sy + action[1]
                else:   #走到边缘了，仍在原地
                    next_sx = sx
                    next_sy = sy

                state_int = self.point_to_int((sx, sy))
                action_int = self.actions.index(action)
                next_state_int = self.point_to_int((next_sx, next_sy))
                reward = self.reward(next_state_int) #r(s')
                #trajectory.append((state_int, action_int, reward))  # 轨迹的形式为（（s,a,r'））
                trajectory.append((state_int, action_int, reward, next_state_int)) #轨迹的形式为（（s,a,r',s'））
                sx = next_sx
                sy = next_sy
            trajectories.append(trajectory)
        return trajectories   # 返回形式为列表[[],[]...]

#test
# grid_size = 4
# wind = 0.3
# discount = 0.9
# gw = Gridworld(grid_size,wind,discount)
# n_trajectories = 10
# trajectory_length = 3*grid_size
# a = gw.generate_trajectories(n_trajectories, trajectory_length)
# print(a)