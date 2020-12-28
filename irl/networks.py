import numpy as np
import torch as t
import random


class QuadraticCost(object):  # 二次损失
    @staticmethod
    def fn(a, y):
        return 0.5 * np.linalg.norm(a - y) ** 2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a - y) * sigmoid_prime(z)


class CrossEntropyCost(object):
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))

    @staticmethod
    def delta(z, a, y):
        return (a - y)


class ANN:
    # layers为列表，其长度给出层数，包括输入和输出层，每个元素给出每层神经元数量
    # 例如[3, 5, 2]代表输入参数为3，中间隐藏层有5个神经元，输出2个结果的神经网络
    def __init__(self, layers, cost = CrossEntropyCost):
        self.num_layers = len(layers)
        self.sizes = layers
        self.cost = cost  # CrossEntropyCost（默认）和QuadraticCost
        self._biases = [np.random.randn(y, 1) for y in layers[1:]]
        self._weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    # sigmoid函数的导数
    def _Dsigmoid(self, z):
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    # 计算输出向量
    def calc(self, input):
        for b, w in zip(self._biases, self._weights):
            input = self._sigmoid(np.dot(w, input) + b)
        return input

    # 训练神经网络
    # trainData: 训练集
    # epochs:    训练轮数，对trainData训练多少轮
    # mini_batch_num:      训练子集的个数
    # learning rate：   学习速率，步长
    # testData:   测试集
    def train(self, trainData, epochs, mini_batch_num, learn_rate, testData=None,threshold=1e-1):
        if testData:
            n_test = len(testData)
        n = len(trainData)
        for j in range(epochs):
            if self._loss(trainData) < threshold:
            # if self._accurary(trainData) > 0.95:
                if testData:
                    print("Epoch {0}: test_error = {1} test_accurary = {2}%".format(j-1, self._loss(testData),100 * self._accurary(testData)))
                else:
                    print("Epoch {0}: train_error = {1} train_accurary = {2}%".format(j-1,self._loss(trainData),100*self._accurary(trainData)))
                break
            random.shuffle(trainData)
            mini_batch_size = n//mini_batch_num
            mini_batches = [trainData[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_bat in mini_batches:
                self._update(mini_bat, learn_rate)
            if testData:
                if (j % 1000 == 0):
                    print("Epoch {0}: test_error = {1} test_accurary = {2}%".format(j, self._loss(testData),100*self._accurary(testData)))
            else:
                if (j % 1000 == 0):
                    print("Epoch {0}: train_error = {1} train_accurary = {2}%".format(j,self._loss(trainData),100*self._accurary(trainData)))
        #return self.accurary(trainData)  #训练正确率%

    def _update(self, mini_batch, learn_rate):
        nabla_b = [np.zeros(b.shape) for b in self._biases]
        nabla_w = [np.zeros(w.shape) for w in self._weights]
        for x, y in mini_batch:
            delta_b, delta_w = self._backpropagation(x, y)
            nabla_b = [nb + db for nb, db in zip(nabla_b, delta_b)]
            nabla_w = [nw + dw for nw, dw in zip(nabla_w, delta_w)]
        self._weights = [w - (learn_rate / len(mini_batch)) * nw for w, nw in zip(self._weights, nabla_w)]
        self._biases = [b - (learn_rate / len(mini_batch)) * nb for b, nb in zip(self._biases, nabla_b)]

    # 后向传播算法
    def _backpropagation(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self._biases]
        nabla_w = [np.zeros(w.shape) for w in self._weights]
        # forward
        activation = x
        activations = [x]
        zs = []
        for b, w in zip(self._biases, self._weights):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self._sigmoid(z)
            activations.append(activation)
        # backward
        delta = self.cost.delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        for layer in range(2, self.num_layers):
            z = zs[-layer]
            sp = self._Dsigmoid(z)
            delta = np.dot(self._weights[-layer + 1].transpose(), delta) * sp
            nabla_b[-layer] = delta
            nabla_w[-layer] = np.dot(delta, activations[-layer - 1].transpose())
        return (nabla_b, nabla_w)

    # 计算数据集MSE
    def _loss(self, Data):
        test_results = [(self.calc(x), self._translate(y)) for (x, y) in Data]
        err = [output - result for (output, result) in test_results]
        loss = sum(np.dot(ei.transpose(), ei) for ei in err)/len(Data) #二维1*1矩阵
        loss = np.squeeze(loss) #去掉多余的维数
        return loss

    #计算数据集正确率0~1
    def _accurary(self, Data):
        output_y = [np.argmax(self.calc(x)) for (x, y) in Data]
        Data_y = [np.argmax(y) for (x, y) in Data]
        accurary = np.sum(np.array(output_y)==np.array(Data_y))/len(Data)
        return accurary

    # 如有必要，将结果进行编码，便于和神经网络输出结果相比较
    # 比如经典的手写识别，就需要把数字y转化为向量
    def _translate(self, y):
        return y

    # 将数字转化成向量
    def vectorized_result(j):
        e = np.zeros((10, 1))
        e[j] = 1.0
        return e


def sigmoid(z):
    """The sigmoid function."""
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z) * (1 - sigmoid(z))
