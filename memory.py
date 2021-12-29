import numpy as np
# memory用于储存跑的数据的数组：

class Memory(object):
    def __init__(self, capacity, dims):
        self.capacity = capacity
        # 保存个数MEMORY_CAPACITY，s_dim * 2 + a_dim + 1：分别是两个state，一个action，和一个reward
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        # 将每个transition存入经验池中
        transition = np.hstack((s, a, [r], s_))
        # 指示经验池是否满，用新的memory取代旧的memory
        index = self.pointer % self.capacity
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)  # 随机选索引
        return self.data[indices, :]