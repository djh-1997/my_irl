import math
from itertools import product
import numpy as np
import numpy.random as rn
from irl.mdp.gridworld import Gridworld

class OWObject (object) : #object在objectworld中
    """
    Object in objectworld.
    """
    def __init__(self, inner_colour, outer_colour):
        self.inner_colour = inner_colour
        self.outer_colour = outer_colour
    def __str__(self):
        return "<OWObject (In: {}) (Out: {})>".format(self.inner_colour, self.outer_colour)
class Objectworld(Gridworld):
    """
    Objectworld MDP.
    """
    def __init__(self, grid_size, n_objects, n_colours, wind, discount):
        """
        grid_size: Grid size. int.
        n_objects: Number of objects in the world. int.
        n_colours: Number of colours to colour objects with. int.
        wind: Chance of moving randomly. float.
        discount: MDP discount. float.
        -> Objectworld
        """
        super().__init__(grid_size, wind, discount) #继承gridworld的初始化函数
        self.actions = ((1, 0), (0, 1), (-1, 0), (0, -1), (0, 0))
        self.n_actions = len(self.actions)
        self.n_objects = n_objects
        self.n_colours = n_colours

        # Generate objects.生成目标点
        self.objects = {} #用字典存储目标点和颜色
        for _ in range(self.n_objects):
            obj = OWObject(rn.randint(self.n_colours),
                           rn.randint(self.n_colours))   #目标点的内外颜色

            while True: #随机生成目标点
                x = rn.randint(self.grid_size)
                y = rn.randint(self.grid_size)

                if (x, y) not in self.objects:
                    break

            self.objects[x, y] = obj

        # Preconstruct the transition probability array.
        self.transition_probability = np.array(
            [[[self._transition_probability(i, j, k) #不确定性MDP(_transition_probability),确定性(_determine_transition_probability)
               for k in range(self.n_states)]
              for j in range(self.n_actions)]
             for i in range(self.n_states)])

    def feature_vector(self, i, discrete=True):
        """
        Get the feature vector associated with a state integer.

        i: State int.
        discrete: Whether the feature vectors should be discrete (default True).
            bool.
        -> Feature vector.
        """
        sx, sy = self.int_to_point(i)

        nearest_inner = {}  # colour: distance
        nearest_outer = {}  # colour: distance

        for y in range(self.grid_size): #计算i到各个目标点外部颜色的最近距离
            for x in range(self.grid_size):
                if (x, y) in self.objects:
                    dist = math.hypot((x - sx), (y - sy)) #状态i到object的欧式距离
                    obj = self.objects[x, y]
                    if obj.inner_colour in nearest_inner:
                        if dist < nearest_inner[obj.inner_colour]:
                            nearest_inner[obj.inner_colour] = dist
                    else:
                        nearest_inner[obj.inner_colour] = dist
                    if obj.outer_colour in nearest_outer:
                        if dist < nearest_outer[obj.outer_colour]:
                            nearest_outer[obj.outer_colour] = dist
                    else:
                        nearest_outer[obj.outer_colour] = dist

        # Need to ensure that all colours are represented.确保所有颜色都有值（方便后面计算处理）
        for c in range(self.n_colours):
            if c not in nearest_inner:
                nearest_inner[c] = 0
            if c not in nearest_outer:
                nearest_outer[c] = 0

        if discrete:  #离散特征
            state = np.zeros((2*self.n_colours*self.grid_size,))
            i = 0
            for c in range(self.n_colours):
                for d in range(1, self.grid_size+1):
                    if nearest_inner[c] < d:
                        state[i] = 1
                    i += 1
                    if nearest_outer[c] < d:
                        state[i] = 1
                    i += 1
            assert i == 2*self.n_colours*self.grid_size
            assert (state >= 0).all()
        else: #连续特征  状态为其到最近各目标内外颜色的距离
            # Continuous features.
            state = np.zeros((2*self.n_colours))
            i = 0
            for c in range(self.n_colours):
                state[i] = nearest_inner[c]
                i += 1
                state[i] = nearest_outer[c]
                i += 1
        return state

    def feature_matrix(self, discrete=True):
        """
        Get the feature matrix for this objectworld.

        discrete: Whether the feature vectors should be discrete (default True).
            bool.
        -> NumPy array with shape (n_states, n_states).
        """

        return np.array([self.feature_vector(i, discrete)
                         for i in range(self.n_states)])

    def reward(self, state_int):   #获取奖赏
        """
        Get the reward for a state int.
        state_int: State int.
        -> reward float
        """
        x, y = self.int_to_point(state_int)

        near_c0 = False  #是否离颜色0（红）3以内
        near_c1 = False  #是否离颜色1（绿）2以内
        for (dx, dy) in product(range(-3, 4), range(-3, 4)):  #返回笛卡尔积的元组（-3，-3）到（3，3）
            if 0 <= x + dx < self.grid_size and 0 <= y + dy < self.grid_size:
                if (abs(dx) + abs(dy) <= 3 and
                        (x+dx, y+dy) in self.objects and
                        self.objects[x+dx, y+dy].outer_colour == 0):
                    near_c0 = True
                if (abs(dx) + abs(dy) <= 2 and
                        (x+dx, y+dy) in self.objects and
                        self.objects[x+dx, y+dy].outer_colour == 1):
                    near_c1 = True

        if near_c0 and near_c1:
            return 1
        if near_c0:
            return -1
        return 0

    def generate_trajectories(self, n_trajectories, trajectory_length, policy):
        """
        Generate n_trajectories trajectories with length trajectory_length.
        n_trajectories: Number of trajectories. int.
        trajectory_length: Length of an episode. int.
        policy: Map from state integers to action integers.
        -> [[(state int, action int, reward float)]]
        """

        return super().generate_trajectories(n_trajectories, trajectory_length,policy, True)

    def optimal_policy(self, state_int):
        raise NotImplementedError(
            "Optimal policy is not implemented for Objectworld.")
    def optimal_policy_deterministic(self, state_int):
        raise NotImplementedError(
            "Optimal policy is not implemented for Objectworld.")
