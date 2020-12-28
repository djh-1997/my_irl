import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
a=np.load('g.npy')
b=np.load('rl1.npy')
c=np.load('rl2.npy')
d=np.load('rl1l2.npy')
# 绘图设置
fig = plt.figure()
ax = fig.gca(projection='3d')  # 三维坐标轴
# X和Y的个数要相同
X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Y = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Z1 = a
Z2 = b
Z3 = c
Z4 = d
# meshgrid把X和Y变成平方长度，比如原来都是4，经过meshgrid和ravel之后，长度都变成了16，因为网格点是16个
xx, yy = np.meshgrid(X, Y)  # 网格化坐标
X, Y = xx.ravel(), yy.ravel()  # 矩阵扁平化
# # 设置柱子属性
height = np.zeros_like(Z1)  # 新建全0数组，shape和Z相同，据说是图中底部的位置
width = depth = 1  # 柱子的长和宽
# # 颜色数组，长度和Z一致
c = ['y'] * len(Z1)

# 开始画图，注意本来的顺序是X, Y, Z, width, depth, height，但是那样会导致不能形成柱子，只有柱子顶端薄片，所以Z和height要互换
ax.bar3d(X, Y, height, width, depth, Z1, color=c, shade=True)  # width, depth, height
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('reward_vale')
plt.title("Groundtruth reward")
plt.show()


fig = plt.figure()
ax = fig.gca(projection='3d')  # 三维坐标轴

# 开始画图，注意本来的顺序是X, Y, Z, width, depth, height，但是那样会导致不能形成柱子，只有柱子顶端薄片，所以Z和height要互换
ax.bar3d(X, Y, height, width, depth, Z2, color=c, shade=True)  # width, depth, height
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('reward_vale')
plt.title("L1=10 Recovered reward")
plt.show()
fig = plt.figure()
ax = fig.gca(projection='3d')  # 三维坐标轴

# 开始画图，注意本来的顺序是X, Y, Z, width, depth, height，(python3.7就行，3.5不行）
ax.bar3d(X, Y, height, width, depth, Z3, color=c, shade=True)  # width, depth, height
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('reward_vale')
plt.title("L2=10 Recovered reward")
plt.show()
fig = plt.figure()
ax = fig.gca(projection='3d')  # 三维坐标轴

# 开始画图，注意本来的顺序是X, Y, Z, width, depth, height，但是那样会导致不能形成柱子，只有柱子顶端薄片，所以Z和height要互换
ax.bar3d(X, Y, height, width, depth, Z4, color=c, shade=True)  # width, depth, height
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('reward_vale')
plt.title("Recovered reward")
plt.show()
fig = plt.figure()
ax = fig.gca(projection='3d')  # 三维坐标轴